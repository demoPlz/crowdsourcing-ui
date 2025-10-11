from threading import Thread
import numpy as np
import torch
import queue
import cv2
import os
import time
import base64
from pathlib import Path

from .constants import *

# RealSense device name patterns to avoid
_REALSENSE_BLOCKLIST = (
    "realsense", "real sense", "d4", "depth", "infrared", "stereo module", "motion module"
)

def _v4l2_node_name(idx: int) -> str:
    """Fast sysfs read of V4L2 device name; '' if unknown."""
    try:
        with open(f"/sys/class/video4linux/video{idx}/name", "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""

def _is_webcam_idx(idx: int) -> bool:
    """True if /dev/video{idx} looks like a regular webcam, not a RealSense node."""
    name = _v4l2_node_name(idx)
    if not name:
        # If we can't read the name, allow it and let open/read decide.
        return True
    return not any(term in name for term in _REALSENSE_BLOCKLIST)

def _prep_capture(cap: cv2.VideoCapture, width=640, height=480, fps=None, mjpg=True):
    """Apply low-latency, webcam-friendly settings once at open."""
    # Keep the buffer tiny to minimize latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Many webcams unlock higher modes with MJPG
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(width))
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps:    cap.set(cv2.CAP_PROP_FPS,          float(fps))

class CameraManager():
    def __init__(self):
        # Regular camera capture
        self.cams = {}
        self._capture_threads: dict[str, Thread] = {}
        self._cap_running: bool = False
        self._latest_jpeg: dict[str, str] = {}
        
        # Observation camera streaming
        self._latest_obs_jpeg: dict[str, str] = {}
        self._obs_img_queue: queue.Queue = queue.Queue(maxsize=int(os.getenv("OBS_STREAM_QUEUE", "8")))
        self._obs_img_running: bool = False
        self._obs_img_thread: Thread | None = None
        
        # JPEG quality and calibration
        self._jpeg_quality = int(os.getenv("JPEG_QUALITY", "80"))
        self._undistort_maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._shutting_down = False
        
        # Load calibrations
        self._load_calibrations()

    def _load_calibrations(self):
        """Load camera calibration data for undistortion."""
        calib_dir = Path(__file__).resolve().parent.parent / "calib"
        
        for cam_name, paths in CALIB_PATHS.items():
            try:
                intr_path = calib_dir / paths["intr"]
                if intr_path.exists():
                    intr_data = np.load(str(intr_path))
                    if "map1" in intr_data and "map2" in intr_data:
                        self._undistort_maps[cam_name] = (intr_data["map1"], intr_data["map2"])
                        print(f"📷 Loaded undistort maps for {cam_name}")
            except Exception as e:
                print(f"⚠️ Failed to load calibration for {cam_name}: {e}")

    def init_cameras(self):
        """Open only *webcams* (skip RealSense nodes) once; skip any that fail."""
        self.cams = getattr(self, "cams", {})
        for name, idx in CAM_IDS.items():
            if name in self.cams:
                continue
            if not _is_webcam_idx(idx):
                print(f"⚠️ Skipping {name} (idx={idx}): looks like RealSense")
                continue
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if not cap.isOpened():
                    print(f"❌ Failed to open camera {name} (idx={idx})")
                    continue
                _prep_capture(cap, width=640, height=480, fps=30)
                # Test read
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"❌ Camera {name} opened but can't read frames")
                    cap.release()
                    continue
                self.cams[name] = cap
                print(f"✅ Camera {name} ready (idx={idx}, shape={frame.shape})")
            except Exception as e:
                print(f"❌ Exception opening camera {name} (idx={idx}): {e}")

        # Start background capture workers after all cameras are opened
        if self.cams and not self._cap_running:
            self._start_camera_workers()

    def _capture_worker(self, name: str, cap: cv2.VideoCapture):
        """
        Background loop: capture frames and encode them as JPEG base64 for web streaming.
        """
        # Small sleep to avoid a tight spin when frames aren't available
        backoff = 0.002
        while self._cap_running and cap.isOpened():
            # Prefer grab/retrieve to drop old frames quickly
            ok = cap.grab()
            if ok:
                ok, frame = cap.retrieve()
            else:
                ok, frame = cap.read()
            if ok and frame is not None:
                # ---- Process in worker: resize → BGR2RGB → undistort (if maps) ----
                if frame.shape[1] != 640 or frame.shape[0] != 480:
                    frame_resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
                else:
                    frame_resized = frame
                rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                maps = self._undistort_maps.get(name)
                if maps is not None:
                    m1, m2 = maps
                    rgb = cv2.remap(rgb, m1, m2, interpolation=cv2.INTER_LINEAR)
                # Pre-encode JPEG base64 (string) for zero-cost serving
                self._latest_jpeg[name] = self._encode_jpeg_base64(rgb)
            else:
                time.sleep(backoff)

    def _start_camera_workers(self):
        """
        Spawn one thread per opened camera to capture continuously.
        """
        if self._cap_running:
            return
        self._cap_running = True
        for name, cap in self.cams.items():
            t = Thread(target=self._capture_worker, args=(name, cap), daemon=True, name=f"cap-{name}")
            t.start()
            self._capture_threads[name] = t

    def push_obs_view(self, name: str, img):
        """Enqueue an observation image for background JPEG encoding; drop if queue is full."""
        if img is None:
            return
        try:
            self._obs_img_queue.put_nowait((name, img))
        except queue.Full:
            # Drop frame to avoid backpressure on add_state
            pass
    
    def _start_obs_stream_worker(self):
        """Start the background observation stream worker."""
        if self._obs_img_running:
            return
        self._obs_img_running = True
        self._obs_img_thread = Thread(target=self._obs_stream_worker, daemon=True)
        self._obs_img_thread.start()

    def _stop_obs_stream_worker(self):
        """Stop the observation stream worker."""
        if not self._obs_img_running:
            return
        self._obs_img_running = False
        try:
            self._obs_img_queue.put_nowait(None)
        except Exception:
            pass
        t = self._obs_img_thread
        if t and t.is_alive():
            t.join(timeout=1.5)
        self._obs_img_thread = None

    def _obs_stream_worker(self):
        """Background worker that processes observation images and converts them to JPEG base64."""
        while self._obs_img_running and not self._shutting_down:
            try:
                item = self._obs_img_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            try:
                name, img = item
                rgb = self._to_uint8_rgb(img)
                if rgb is not None:
                    self._latest_obs_jpeg[name] = self._encode_jpeg_base64(rgb)
            except Exception:
                pass
            finally:
                try:
                    self._obs_img_queue.task_done()
                except Exception:
                    pass

    def _snapshot_latest_views(self) -> dict[str, str]:
        """
        Snapshot the latest **JPEG base64 strings** for each camera.
        We copy dict entries to avoid referencing a dict being mutated by workers.
        """
        out: dict[str, str] = {}
        for name in ("left", "right", "front", "perspective"):
            s = self._latest_jpeg.get(name)
            if s is not None:
                out[name] = s

        # Include latest observation camera previews if available
        for name in ("obs_main", "obs_wrist"):
            s = self._latest_obs_jpeg.get(name)
            if s is not None:
                out[name] = s
        return out

    def _encode_jpeg_base64(self, img_rgb: np.ndarray, quality: int | None = None) -> str:
        """Encode RGB image as JPEG base64 string."""
        if quality is None:
            quality = self._jpeg_quality
        # OpenCV expects BGR for JPEG encoding
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return ""
        b64 = base64.b64encode(buf).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _to_uint8_rgb(self, arr) -> np.ndarray | None:
        """Convert various array formats to uint8 RGB."""
        if arr is None:
            return None
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().to("cpu").numpy()
        if not isinstance(arr, np.ndarray):
            return None
        # Accept HxWx3 or 3xHxW
        if arr.ndim != 3:
            return None
        if arr.shape[0] == 3 and arr.shape[2] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            try:
                maxv = float(np.nanmax(arr))
            except Exception:
                maxv = 255.0
            if arr.dtype.kind in "fc" and maxv <= 1.0:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        # Expect RGB already; do not swap channels here.
        return arr if arr.shape[2] == 3 else None

    def shutdown(self):
        """Clean shutdown of all camera resources."""
        self._shutting_down = True
        
        # Stop observation stream worker
        self._stop_obs_stream_worker()
        
        # Stop camera capture workers
        self._cap_running = False
        for t in self._capture_threads.values():
            if t.is_alive():
                t.join(timeout=1.0)
        
        # Release camera resources
        for name, cap in self.cams.items():
            try:
                cap.release()
                print(f"📷 Released camera {name}")
            except Exception as e:
                print(f"⚠️ Error releasing camera {name}: {e}")
        
        self.cams.clear()
        self._capture_threads.clear()