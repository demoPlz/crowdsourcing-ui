"""
Webcam Management Module

Handles webcam capture, background encoding, and frame streaming for the crowd interface.
Manages V4L2 camera devices with RealSense filtering and undistortion.
"""

import os
import time
import base64
from threading import Thread
from pathlib import Path

import cv2
import numpy as np


# RealSense device name patterns to skip (we only want regular webcams)
_REALSENSE_BLOCKLIST = (
    "realsense", "real sense", "d4", "depth", "infrared", "stereo module", "motion module"
)


def _v4l2_node_name(idx: int) -> str:
    """Fast sysfs read of V4L2 device name; '' if unknown."""
    try:
        with open(f"/sys/class/video4linux/video{idx}/name", "r", encoding="utf-8") as f:
            return f.read().strip().lower()
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


class WebcamManager:
    """
    Manages webcam capture and background JPEG encoding for streaming.
    
    Responsibilities:
    - Open and initialize V4L2 webcams (filtering out RealSense devices)
    - Background capture threads with continuous frame grabbing
    - JPEG encoding with base64 data URLs for web streaming
    - Undistortion map application
    - Latest frame caching for low-latency access
    
    Attributes:
        cams: Dict of camera name -> cv2.VideoCapture
        cam_ids: Dict of camera name -> V4L2 device index
    """
    
    def __init__(
        self,
        cam_ids: dict[str, int],
        undistort_maps: dict[str, tuple] | None = None,
        jpeg_quality: int | None = None
    ):
        """
        Initialize webcam manager.
        
        Args:
            cam_ids: Mapping of camera names to V4L2 device indices (e.g., {"front": 18, "left": 4})
            undistort_maps: Optional dict of camera name -> (map1, map2) for cv2.remap undistortion
            jpeg_quality: JPEG encoding quality (1-100), defaults to env JPEG_QUALITY or 80
        """
        self.cam_ids = cam_ids
        self._undistort_maps = undistort_maps or {}
        self._jpeg_quality = int(jpeg_quality or os.getenv("JPEG_QUALITY", "80"))
        
        # Camera state
        self.cams: dict[str, cv2.VideoCapture] = {}
        
        # Background capture state
        self._cap_threads: dict[str, Thread] = {}
        self._cap_running: bool = False
        self._latest_jpeg: dict[str, str] = {}
    
    def init_cameras(self):
        """
        Open only *webcams* (skip RealSense nodes) once; skip any that fail.
        Automatically starts background capture workers after cameras are opened.
        """
        for name, idx in self.cam_ids.items():
            # Only attempt indices that look like webcams
            if not _is_webcam_idx(idx):
                print(f"⏭️  skipping '{name}' (/dev/video{idx}) - not a webcam")
                continue
            
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                print(f"⚠️  camera '{name}' (id {idx}) could not be opened")
                continue
            
            # One-time efficiency settings
            _prep_capture(cap, width=640, height=480, fps=None, mjpg=True)
            
            try:
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
            except Exception:
                pass
            
            # Verify we can actually read one frame
            ok, _ = cap.read()
            if not ok:
                cap.release()
                print(f"⚠️  camera '{name}' (id {idx}) opens but won't deliver frames")
                continue
            
            self.cams[name] = cap
            print(f"✓ Camera '{name}' opened successfully (/dev/video{idx})")
        
        # Start background capture workers after all cameras are opened
        if self.cams and not self._cap_running:
            self._start_camera_workers()
    
    def _start_camera_workers(self):
        """
        Spawn one thread per opened camera to capture continuously.
        """
        if self._cap_running:
            return
        self._cap_running = True
        for name, cap in self.cams.items():
            t = Thread(target=self._capture_worker, args=(name, cap), daemon=True)
            t.start()
            self._cap_threads[name] = t
    
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
                    map1, map2 = maps
                    rgb = cv2.remap(rgb, map1, map2, cv2.INTER_LINEAR)
                # Pre-encode JPEG base64 (string) for zero-cost serving
                self._latest_jpeg[name] = self.encode_jpeg_base64(rgb)
            else:
                time.sleep(backoff)
    
    def snapshot_latest_views(self) -> dict[str, str]:
        """
        Snapshot the latest **JPEG base64 strings** for each camera.
        We copy dict entries to avoid referencing a dict being mutated by workers.
        
        Returns:
            Dict of camera name -> JPEG data URL (base64 encoded)
        """
        out: dict[str, str] = {}
        for name in self.cams.keys():
            s = self._latest_jpeg.get(name)
            if s is not None:
                out[name] = s
        return out
    
    def encode_jpeg_base64(self, img_rgb: np.ndarray, quality: int | None = None) -> str:
        """
        Encode an RGB image to a base64 JPEG data URL.
        
        Args:
            img_rgb: RGB image as numpy array (HxWx3)
            quality: Optional JPEG quality override (1-100)
        
        Returns:
            Data URL string: "data:image/jpeg;base64,..."
        """
        q = int(self._jpeg_quality if quality is None else quality)
        if not img_rgb.flags["C_CONTIGUOUS"]:
            img_rgb = np.ascontiguousarray(img_rgb)
        # OpenCV imencode expects BGR
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return ""
        b64 = base64.b64encode(buf).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    
    def stop(self):
        """Stop all capture workers and release cameras."""
        self._cap_running = False
        # Wait for threads to finish
        for thread in self._cap_threads.values():
            if thread.is_alive():
                thread.join(timeout=1.0)
        # Release all cameras
        for cap in self.cams.values():
            cap.release()
        self.cams.clear()
        self._cap_threads.clear()
        self._latest_jpeg.clear()
