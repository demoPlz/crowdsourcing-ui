"""
Observation Stream Manager Module

Handles observation camera live preview streaming with background JPEG encoding.
Manages conversion and encoding of observation images (cam_main, cam_wrist) for web streaming.
"""

import queue
from threading import Thread

import numpy as np
import torch


class ObservationStreamManager:
    """
    Manages observation camera live preview streaming with background encoding.
    
    Responsibilities:
    - Background JPEG encoding worker thread
    - Image format conversion (torch.Tensor/ndarray → uint8 RGB)
    - Queue-based image processing with backpressure handling
    - Latest observation JPEG caching for obs_main and obs_wrist
    
    Attributes:
        _latest_obs_jpeg: Dict of observation name → base64 JPEG data URL
        _obs_img_queue: Queue for background encoding tasks
    """
    
    def __init__(
        self,
        encoder_func,
        queue_maxsize: int | None = None
    ):
        """
        Initialize observation stream manager.
        
        Args:
            encoder_func: Function to encode RGB image to base64 JPEG (from WebcamManager)
            queue_maxsize: Max queue size for background encoding (defaults to env OBS_STREAM_QUEUE or 8)
        """
        self._encoder_func = encoder_func
        
        # Queue and worker state
        import os
        self._obs_img_queue: queue.Queue = queue.Queue(maxsize=int(queue_maxsize or os.getenv("OBS_STREAM_QUEUE", "8")))
        self._obs_img_running: bool = False
        self._obs_img_thread: Thread | None = None
        
        # Latest encoded images
        self._latest_obs_jpeg: dict[str, str] = {}
        
        # Start background worker
        self._start_obs_stream_worker()
    
    def _start_obs_stream_worker(self):
        """Start the background JPEG encoding worker thread."""
        if self._obs_img_running:
            return
        self._obs_img_running = True
        self._obs_img_thread = Thread(target=self._obs_stream_worker, daemon=True)
        self._obs_img_thread.start()
    
    def _to_uint8_rgb(self, arr) -> np.ndarray | None:
        """
        Convert various image formats to uint8 RGB numpy array.
        
        Handles:
        - torch.Tensor (any device) → numpy
        - Float [0,1] or [0,255] ranges
        - Channel-first (3xHxW) or channel-last (HxWx3)
        - Non-contiguous arrays
        
        Args:
            arr: Image as torch.Tensor or np.ndarray
        
        Returns:
            RGB uint8 numpy array (HxWx3) or None if invalid
        """
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
    
    def _obs_stream_worker(self):
        """
        Background worker loop: dequeue images and encode them as JPEG.
        """
        while True:
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
                    self._latest_obs_jpeg[name] = self._encoder_func(rgb)
            except Exception:
                pass
            finally:
                try:
                    self._obs_img_queue.task_done()
                except Exception:
                    pass
    
    def push_obs_view(self, name: str, img):
        """
        Enqueue an observation image for background JPEG encoding.
        Drops frame if queue is full to avoid backpressure.
        
        Args:
            name: Observation camera name (e.g., "obs_main", "obs_wrist")
            img: Image as torch.Tensor or np.ndarray
        """
        if img is None:
            return
        try:
            self._obs_img_queue.put_nowait((name, img))
        except queue.Full:
            # Drop frame to avoid backpressure on add_state
            pass
    
    def get_latest_obs_jpeg(self) -> dict[str, str]:
        """
        Get the latest encoded observation images.
        
        Returns:
            Dict of observation name → JPEG base64 data URL
        """
        return self._latest_obs_jpeg.copy()
    
    def stop(self):
        """Stop the background encoding worker."""
        self._obs_img_running = False
        # Signal worker to stop
        try:
            self._obs_img_queue.put_nowait(None)
        except queue.Full:
            pass
        # Wait for thread to finish
        if self._obs_img_thread and self._obs_img_thread.is_alive():
            self._obs_img_thread.join(timeout=1.0)
