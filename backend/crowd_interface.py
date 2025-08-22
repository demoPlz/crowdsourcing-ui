from __future__ import annotations

REQUIRED_RESPONSES_PER_STATE = 3

CAM_IDS = {
    "front":       18,   # change indices / paths as needed
    "left":        4,
    "right":       0,
    "perspective": 2,
}

JOINT_NAMES = [
    "joint_0", "joint_1", "joint_2",
    "joint_3", "joint_4", "joint_5",
    "left_carriage_joint"
]

# Per-camera calibration file paths (extend as you calibrate more cams)
# Use T_three from extrinsics (camera world for Three.js) and map1/map2 from intrinsics to undistort.
CALIB_PATHS = {
    "front": {
        "intr": "calib/intrinsics_front_1_640x480.npz",
        "extr": "calib/extrinsics_front_1.npz",
    },
    "left": {
        "intr": "calib/intrinsics_left_2_640x480.npz",
        "extr": "calib/extrinsics_left_2.npz",
    },
    "right": {
        "intr": "calib/intrinsics_right_3_640x480.npz",
        "extr": "calib/extrinsics_right_3.npz",
    },
    "perspective": {
        "intr": "calib/intrinsics_perspective_4_640x480.npz",
        "extr": "calib/extrinsics_perspective_4.npz",
    },
}

import cv2
import numpy as np
import os
import time
import torch
import base64

from flask import Flask, jsonify
from flask_cors import CORS
from flask import request
from pathlib import Path
from threading import Thread, Lock
from collections import deque
from math import cos, sin
from math import cos, sin
import json
import base64

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.control_utils import sanity_check_dataset_robot_compatibility, sanity_check_dataset_name

_REALSENSE_BLOCKLIST = (
    "realsense", "real sense", "d4", "depth", "infrared", "stereo module", "motion module"
)

### HELPERS

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

class CrowdInterface():
    '''
    Sits between the frontend and the backend
    '''
    def __init__(self, required_responses_per_state= REQUIRED_RESPONSES_PER_STATE):

        self.states = deque(maxlen=4)
        self.cams = {}
        self.latest_goal = None
        self.goal_lock = Lock()
        self._gripper_motion = 1  # Initialize gripper motion

        self.robot_is_moving = False
        
        # N responses pattern
        self.required_responses_per_state = required_responses_per_state
        self.pending_states = {}  # state_id -> {state: dict, responses_received: int, timestamp: float}
        self.next_state_id = 0
        self.state_lock = Lock()  # Protects pending_states and next_state_id
        
        # Track which state was served to which user session
        self.served_states = {}  # session_id -> state_id (most recently served state to this session)

        # Dataset
        self.dataset = None

        # Task
        self.task = None

        # Background capture state
        self._cap_threads: dict[str, Thread] = {}
        self._cap_running: bool = False
        self._latest_raw: dict[str, np.ndarray] = {}
        self._latest_ts: dict[str, float] = {}
        self._latest_proc: dict[str, np.ndarray] = {}
        self._latest_jpeg: dict[str, str] = {}
        # JPEG quality for base64 encoding (override with env JPEG_QUALITY)
        self._jpeg_quality = int(os.getenv("JPEG_QUALITY", "80"))

        # Will be filled by _load_calibrations()
        self._undistort_maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._camera_models: dict[str, dict] = {}
        self._camera_poses = self._load_calibrations()


        # Precompute immutable views and camera poses to avoid per-tick allocations
        H, W = 64, 64
        pink = np.full((H, W, 3), 255, dtype=np.uint8)  # one-time NumPy buffer
        blank = pink.tolist()                           # one-time JSON-serializable view
        # Reuse the same object for all cameras (read-only downstream)
        self._blank_views = {
            "left":        blank,
            "right":       blank,
            "front":       blank,
            "perspective": blank,
        }
    
    ### ---Dataset Management---
    def init_dataset(self, 
                     cfg,
                     robot):
        """Intialize dataset for data collection policy training"""
        if cfg.resume:
            self.dataset = LeRobotDataset(
                cfg.data_collection_policy_repo_id,
                root=cfg.root
            )
            self.dataset.start_image_writer(
                num_processes=cfg.num_image_writer_processes,
                num_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )
            sanity_check_dataset_robot_compatibility(self.dataset, robot, cfg.fps, cfg.video)

        else:
            sanity_check_dataset_name(cfg.data_collection_policy_repo_id, cfg.policy)
            self.dataset = LeRobotDataset.create(
                cfg.data_collection_policy_repo_id,
                cfg.fps,
                root=cfg.root,
                robot=robot,
                use_videos=cfg.video,
                image_writer_processes=cfg.num_image_writer_processes,
                image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )

        self.task = cfg.single_task

    def set_dataset_reference(self, dataset, task: str):
        """Set the dataset reference and task for completed state recording"""
        self.dataset = dataset
        self.task = task

    ### ---Camera Management---
    def init_cameras(self):
        """Open only *webcams* (skip RealSense nodes) once; skip any that fail."""
        self.cams = getattr(self, "cams", {})
        for name, idx in CAM_IDS.items():
            # Only attempt indices that look like webcams
            if not _is_webcam_idx(idx):
                print(f"⏭️  skipping '{name}' (/dev/video{idx}) — not a webcam")
                continue

            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                print(f"⚠️  camera “{name}” (id {idx}) could not be opened")
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
                print(f"⚠️  camera “{name}” (id {idx}) opens but won't deliver frames")
                continue

            self.cams[name] = cap
            print(f"✓ Camera '{name}' opened successfully (/dev/video{idx})")

        # Start background capture workers after all cameras are opened
        if self.cams and not self._cap_running:
            self._start_camera_workers()


    def cleanup_cameras(self):
        """Close all cameras"""
        # Stop background workers
        if self._cap_running:
            self._cap_running = False
            for t in list(self._cap_threads.values()):
                try:
                    t.join(timeout=0.5)
                except Exception:
                    pass
            self._cap_threads.clear()
        self._latest_raw.clear()
        self._latest_ts.clear()
        self._latest_proc.clear()
        self._latest_jpeg.clear()

        for cap in getattr(self, "cams", {}).values():
            try:
                cap.release()
            except Exception:
                pass
        self.cams = {}


    def get_views(self) -> dict[str, list]:
        """Return an RGB image dict from available webcams.
        Uses grab() → retrieve() pattern for near-simultaneous multi-cam capture."""
        if not hasattr(self, "cams"):
            self.cams = {}

        order = ("left", "right", "front", "perspective")
        # 1) grab from all first (non-blocking dequeue)
        for name in order:
            if name in self.cams:
                self.cams[name].grab()

        # 2) retrieve, convert, downscale
        views: dict[str, list] = {}
        for name in order:
            if name not in self.cams:
                continue
            frame = self._grab_frame(self.cams[name], size=(640, 480))
            # Apply per-camera undistortion (if intrinsics provided)
            maps = self._undistort_maps.get(name)
            if frame is not None and maps is not None:
                m1, m2 = maps
                frame = cv2.remap(frame, m1, m2, interpolation=cv2.INTER_LINEAR)
            if frame is not None:
                views[name] = frame.tolist()
        return views


    def _grab_frame(self, cap, size=(64, 64)) -> np.ndarray | None:
        # retrieve() after grab(); falls back to read() if needed
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
        # Resize then convert to RGB; INTER_AREA is efficient for downscale
        if size is not None:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    def _grab_frame_raw(self, cap) -> np.ndarray | None:
        """
        Retrieve a raw BGR frame at native resolution with no resize,
        color conversion, or undistortion. Used for high-rate capture.
        """
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
        return frame
    
    def _capture_worker(self, name: str, cap: cv2.VideoCapture):
        """
        Background loop: keep the latest raw BGR frame in self._latest_raw[name].
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
                # Keep raw for debug/inspection (BGR, native size)
                self._latest_raw[name] = frame
                ts = time.time()
                self._latest_ts[name] = ts
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
                # Atomic pointer swap to latest processed frame (RGB, 640x480)
                self._latest_proc[name] = rgb
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
            t = Thread(target=self._capture_worker, args=(name, cap), daemon=True)
            t.start()
            self._cap_threads[name] = t

    def _snapshot_latest_views(self) -> dict[str, np.ndarray]:
        """
        Snapshot the latest **JPEG base64 strings** for each camera.
        We copy dict entries to avoid referencing a dict being mutated by workers.
        """
        out: dict[str, str] = {}
        for name in ("left", "right", "front", "perspective"):
            s = self._latest_jpeg.get(name)
            if s is not None:
                out[name] = s
        return out

    def _get_views_np(self) -> dict[str, np.ndarray]:
        """
        Return raw NumPy frames (BGR, native size) without resize/undistort.
        Use this in the high-rate control loop; convert on /api/get-state.
        """
        if not hasattr(self, "cams"):
            self.cams = {}

        order = ("left", "right", "front", "perspective")
        # 1) grab from all first (non-blocking dequeue)
        for name in order:
            if name in self.cams:
                self.cams[name].grab()

        # 2) retrieve, convert, (optional) undistort
        views: dict[str, np.ndarray] = {}
        for name in order:
            if name not in self.cams:
                continue
            frame = self._grab_frame_raw(self.cams[name])
            if frame is not None:
                views[name] = frame
        return views

    def _state_to_json(self, state: dict) -> dict:
        """
        Convert an internal state into a JSON-serializable dict.
        Views are already JPEG base64 strings produced in background workers,
        so we just pass them through. (If an ndarray sneaks in, encode it.)
        
        Note: observations and actions are not included in the frontend state
        as they're not needed there - only state_id is used for correspondence.
        """
        if not state:
            return {}
        out = dict(state)  # shallow copy; we replace 'views'
        
        # Handle views (convert any arrays to base64 strings)
        raw_views = state.get("views") or {}
        safe_views: dict[str, str] = {}
        for name, frame in raw_views.items():
            if isinstance(frame, str):
                safe_views[name] = frame
            elif isinstance(frame, np.ndarray):
                # Fallback: encode on the fly if we ever got an array here
                safe_views[name] = self._encode_jpeg_base64(frame)
            else:
                # Unknown type -> drop or stringify minimally
                safe_views[name] = ""
        out["views"] = safe_views
        
        return out
    
    def _encode_jpeg_base64(self, img_rgb: np.ndarray, quality: int | None = None) -> str:
        """
        Encode an RGB image to a base64 JPEG data URL.
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
    
    # --- State Management ---
    def add_state(self, joint_positions: dict, gripper_motion: int = None, obs_dict: dict[str, torch.Tensor] = None):
        if gripper_motion is not None:
            self._gripper_motion = int(gripper_motion)

        # Cheap, explicit cast of the 7 scalars to built-in floats
        jp = {k: float(v) for k, v in joint_positions.items()}

        # Frontend state (lightweight, no observations/actions)
        frontend_state = {
            "joint_positions": jp,
            "views": self._snapshot_latest_views(),
            "camera_poses": self._camera_poses,  # reuse precomputed poses
            "camera_models": self._camera_models,  # per-camera intrinsics for Three.js
            "gripper": self._gripper_motion,
            "controls": ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper'],
        }
        
        # Add to both old system (for backward compatibility) and new N responses system
        self.states.append(frontend_state)
        
        # Add to pending states for N responses pattern
        with self.state_lock:
            state_id = self.next_state_id
            self.next_state_id += 1
            self.pending_states[state_id] = {
                "state": frontend_state.copy(),  # Frontend state (lightweight)
                "observations": obs_dict,  # Keep observations for dataset creation
                "actions": [],  # Will collect action responses here
                "responses_received": 0,
                "timestamp": time.time()
            }
        
        print(f"🟢 State {state_id} added. Pending states: {len(self.pending_states)}")
        # print(f"🟢 Joint positions: {joint_positions}")
        # print(f"🟢 Gripper: {self._gripper_motion}")
    
    def get_latest_state(self, session_id: str = "default") -> dict:
        """
        Get a state that needs responses with movement-aware prioritization:
        - When robot is NOT moving: prioritize LATEST state for immediate execution
        - When robot is moving: prioritize OLDEST state for systematic collection
        
        Args:
            session_id: Unique identifier for the user session (for tracking which state was served)
        """
        with self.state_lock:
            if not self.pending_states:
                return {}
            
            if self.robot_is_moving is False:
                # Robot not moving - prioritize LATEST state for immediate execution
                latest_state_id = max(self.pending_states.keys(), 
                                    key=lambda sid: self.pending_states[sid]["timestamp"])
                state = self.pending_states[latest_state_id]["state"].copy()
                
                # Track which state was served to this session
                self.served_states[session_id] = latest_state_id
                
                # Include state_id in the response so frontend can send it back
                state["state_id"] = latest_state_id
                
                print(f"🎯 Robot stationary - serving LATEST state {latest_state_id} to session {session_id} for immediate execution")
                return state
            else:
                # Robot moving - prioritize OLDEST state for systematic data collection
                oldest_state_id = min(self.pending_states.keys(), 
                                    key=lambda sid: self.pending_states[sid]["timestamp"])
                state = self.pending_states[oldest_state_id]["state"].copy()
                
                # Track which state was served to this session
                self.served_states[session_id] = oldest_state_id
                
                # Include state_id in the response so frontend can send it back
                state["state_id"] = oldest_state_id
                
                print(f"🔍 Robot moving - serving OLDEST state {oldest_state_id} to session {session_id} for data collection")
                return state
    
    def record_response(self, response_data: dict, session_id: str = "default") -> bool:
        """
        Record a response for a specific state.
        Returns True if this completes the required responses for a state.
        
        Args:
            response_data: The response data containing state_id and action data
            session_id: The user session that submitted the response
        """
        with self.state_lock:
            if not self.pending_states:
                print("⚠️  No pending states available to record response")
                return False
            
            # Try to get state_id from response data first
            state_id = response_data.get("state_id")
            if state_id is not None:
                print(f"✅ Received state_id {state_id} from frontend for session {session_id}")
            
            # Validate that the state still exists
            if state_id not in self.pending_states:
                print(f"⚠️  State {state_id} no longer exists in pending states")
                return False
            
            pending_info = self.pending_states[state_id]
            pending_info["responses_received"] += 1
            
            # Extract joint positions and gripper action from response
            joint_positions = response_data.get("joint_positions", {})
            gripper_action = response_data.get("gripper", 0)
            
            # Assemble goal_positions like in teleop_step_crowd
            # Convert joint positions dict to ordered list matching JOINT_NAMES
            goal_positions = []
            for joint_name in JOINT_NAMES:
                goal_positions.append(float(joint_positions.get(joint_name, 0.0)[0]))
            
            # Handle gripper like in teleop_step_crowd: set to 0.044 or 0.0 based on sign
            goal_positions[-1] = 0.044 if gripper_action > 0 else 0.0
            goal_positions = torch.tensor(goal_positions, dtype=torch.float32)
            
            # Store this action response in the same format as teleop_step_crowd
            pending_info["actions"].append(goal_positions)

            print(f"🔔 Response recorded for state {state_id} from session {session_id} ({pending_info['responses_received']}/{self.required_responses_per_state})")
            
            # Check if we've received enough responses
            if pending_info["responses_received"] >= self.required_responses_per_state:
                # Stack all action responses into final action tensor
                # This will have shape [REQUIRED_RESPONSES_PER_STATE, action_dim]
                all_actions = torch.stack(pending_info["actions"], dim=0)
                
                # Create the complete frame for dataset.add_frame() 
                # Format matches exactly what teleop_step_crowd produces
                completed_state = {
                    **pending_info["observations"],  # observations dict
                    "action": all_actions,           # action tensor with all crowd responses
                    "task": self.task if self.task else "crowdsourced_task"
                }
                
                self.dataset.add_frame(completed_state)
                
                # Remove from pending states
                del self.pending_states[state_id]
                print(f"✅ State {state_id} completed, removed from pending. Remaining: {len(self.pending_states)}")
                
                # Clean up served states tracking for this completed state
                sessions_to_clean = [sid for sid, served_state_id in self.served_states.items() 
                                   if served_state_id == state_id]
                for sid in sessions_to_clean:
                    del self.served_states[sid]
                
                return True
            
            return False
    
    def get_pending_states_info(self) -> dict:
        """Get information about pending states for debugging/monitoring"""
        with self.state_lock:
            return {
                "total_pending": len(self.pending_states),
                "required_responses_per_state": self.required_responses_per_state,
                "states_info": {
                    state_id: {
                        "responses_received": info["responses_received"],
                        "responses_needed": self.required_responses_per_state - info["responses_received"],
                        "age_seconds": time.time() - info["timestamp"]
                    }
                    for state_id, info in self.pending_states.items()
                }
            }
    
    # --- Goal Management ---
    def submit_goal(self, goal_data: dict):
        """Submit a new goal from the frontend"""
        if not self.robot_is_moving:
            self.latest_goal = goal_data
        # print(f"🔔 Goal received: {goal_data}")
    
    def get_latest_goal(self) -> dict | None:
        """Get and clear the latest goal (for robot loop to consume)"""
        goal = self.latest_goal
        self.latest_goal = None
        return goal
    
    def has_pending_goal(self) -> bool:
        """Check if there's a pending goal"""
        return self.latest_goal is not None
    
    # --- Helper Methods ---

    def _make_camera_poses(self) -> dict[str, list]: #Fallback
        def euler_pose(x: float, y: float, z: float,
               roll: float, pitch: float, yaw: float) -> list[list[float]]:
            """
            Build a 4x4 **world** matrix (row-major list-of-lists) from
            T = Trans(x,y,z) · Rz(yaw) · Ry(pitch) · Rx(roll)
            """
            cr, sr = cos(roll),  sin(roll)
            cp, sp = cos(pitch), sin(pitch)
            cy, sy = cos(yaw),   sin(yaw)

            # column-major rotation
            Rrow = np.array([
                [ cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                [ sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                [   -sp,              cp*sr,              cp*cr]
            ])

            T = np.eye(4)
            T[:3, :3] = Rrow.T
            T[:3,  3] = [x, y, z]
            return T.tolist()
        
        return {
            #           x     y     z     roll   pitch   yaw
            "front_pose":       euler_pose(0.2, -1.0, 0.15, -np.pi/2, 0.0, 0.0),
            "left_pose":        euler_pose(0.2, -1.0, 0.15, -np.pi/2, 0.0, 0.0),
            "right_pose":       euler_pose(0.2,  1.0, 0.15, np.pi/2, 0.0, np.pi),
            "perspective_pose": euler_pose(1.3,  1.0, 1.0, np.pi/4, -np.pi/4, -3*np.pi/4),
        }
    
    def _load_calibrations(self) -> dict[str, list]:
        """
        Load per-camera extrinsics (→ camera_poses) and intrinsics (→ undistortion  Knew for projection).
        Falls back to placeholder poses for any camera missing calibrations.
        """
        poses = self._make_camera_poses()  # start with fallbacks
        self._undistort_maps = {}
        self._camera_models = {}

        # Base directory for optional manual overrides: ../calib/manual_calibration_{name}.json
        base_dir = Path(__file__).resolve().parent
        manual_dir = (base_dir / ".." / "calib").resolve()

        for name, paths in CALIB_PATHS.items():
            if not paths:
                continue

            # ---- Load extrinsics → camera pose ----
            extr = paths.get("extr")
            if extr and os.path.exists(extr):
                try:
                    data = np.load(extr, allow_pickle=True)
                    if "T_three" in data:
                        M = np.asarray(data["T_three"], dtype=np.float64)
                    elif "T_base_cam" in data:
                        # Convert OpenCV cam (Z forward) to Three.js cam (looks -Z)
                        T = np.asarray(data["T_base_cam"], dtype=np.float64)
                        Rflip = np.diag([1.0, -1.0, -1.0])
                        M = np.eye(4, dtype=np.float64)
                        M[:3, :3] = T[:3, :3] @ Rflip
                        M[:3,  3] = T[:3,  3]
                    else:
                        M = None
                    if M is not None:
                        poses[f"{name}_pose"] = M.tolist()
                        print(f"✓ loaded extrinsics for '{name}' from {extr}")
                except Exception as e:
                    print(f"⚠️  failed to load extrinsics for '{name}' ({extr}): {e}")

            # ---- Load intrinsics → undistortion maps  Knew for projection ----
            intr = paths.get("intr")
            if intr and os.path.exists(intr):
                try:
                    idata = np.load(intr, allow_pickle=True)
                    W = int(idata["width"])
                    H = int(idata["height"])
                    # Prefer rectified Knew (matches undistorted frames)
                    Knew = np.asarray(idata["Knew"], dtype=np.float64)
                    # Optional: precomputed undistort maps
                    if "map1" in idata.files and "map2" in idata.files:
                        self._undistort_maps[name] = (idata["map1"], idata["map2"])
                        rectified = True
                        print(f"✓ loaded undistort maps for '{name}' from {intr}")
                    else:
                        rectified = False
                    # Expose per-camera intrinsics to the frontend
                    self._camera_models[name] = {
                        "model": "pinhole",
                        "rectified": rectified,
                        "width": W,
                        "height": H,
                        "Knew": Knew.tolist(),
                        # (optionally include original K/D if you want)
                        # "K": np.asarray(idata["K"], dtype=np.float64).tolist(),
                        # "D": np.asarray(idata["D"], dtype=np.float64).ravel().tolist(),
                    }
                    print(f"✓ loaded intrinsics (Knew {W}x{H}) for '{name}' from {intr}")
                except Exception as e:
                    print(f"⚠️  failed to load intrinsics for '{name}' ({intr}): {e}")

            # ---- Manual override (JSON) if present ----
            # File: ../calib/manual_calibration_{name}.json
            try:
                manual_path = manual_dir / f"manual_calibration_{name}.json"
                if manual_path.exists():
                    with open(manual_path, "r", encoding="utf-8") as f:
                        mcal = json.load(f)
                    intr_m = (mcal or {}).get("intrinsics") or {}
                    extr_m = (mcal or {}).get("extrinsics") or {}
                    # Validate presence of fields we expect
                    if "T_three" in extr_m and isinstance(extr_m["T_three"], list):
                        poses[f"{name}_pose"] = extr_m["T_three"]
                        print(f"✓ applied MANUAL extrinsics for '{name}' from {manual_path}")
                    if all(k in intr_m for k in ("width", "height", "Knew")):
                        # Preserve existing 'rectified' flag if any, otherwise False
                        prev_rect = self._camera_models.get(name, {}).get("rectified", False)
                        self._camera_models[name] = {
                            "model": "pinhole",
                            "rectified": prev_rect,
                            "width": int(intr_m["width"]),
                            "height": int(intr_m["height"]),
                            "Knew": intr_m["Knew"],
                        }
                        print(f"✓ applied MANUAL intrinsics for '{name}' from {manual_path}")
            except Exception as e:
                print(f"⚠️  failed to apply manual calibration for '{name}': {e}")

        return poses

    def is_recording(self):
        return len(self.pending_states) > 0

def create_flask_app(crowd_interface: CrowdInterface) -> Flask:
    """Create and configure Flask app with the crowd interface"""
    app = Flask(__name__)
    CORS(app)
    
    @app.route("/api/get-state")
    def get_state():
        current_time = time.time()
        
        # Generate or retrieve session ID from request headers or IP
        session_id = request.headers.get('X-Session-ID', request.remote_addr)
        
        state = crowd_interface.get_latest_state(session_id)
        # print(f"🔍 Flask route /api/get-state called at {current_time}")
        # print(f"🔍 Pending states: {len(crowd_interface.pending_states)}")
        payload = crowd_interface._state_to_json(state)
        
        # Add hardcoded prompt text
        payload["prompt"] = "Pick up the red block."
        
        response = jsonify(payload)
        # Prevent caching
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    
    @app.route("/api/test")
    def test():
        return jsonify({"message": "Flask server is working", "states_count": len(crowd_interface.states)})
    
    @app.route("/api/submit-goal", methods=["POST"])
    def submit_goal():
        data = request.get_json(force=True, silent=True) or {}
        
        # Generate or retrieve session ID from request headers or IP
        session_id = request.headers.get('X-Session-ID', request.remote_addr)
        
        crowd_interface.submit_goal(data)
        
        # Record this as a response to the correct state for this session
        # The frontend now includes state_id in the request data
        crowd_interface.record_response(data, session_id)
        
        return jsonify({"status": "ok"})
    
    @app.route("/api/pending-states-info")
    def pending_states_info():
        """Debug endpoint to see pending states information"""
        info = crowd_interface.get_pending_states_info()
        return jsonify(info)
    
    @app.route("/api/save-calibration", methods=["POST"])
    def save_calibration():
        """
        Save manual calibration to ../calib/manual_calibration_{camera}.json
        Also updates the in-memory camera models/poses so the user immediately sees results.
        Expected JSON:
        {
          "camera": "front",
          "intrinsics": {"width": W, "height": H, "Knew": [[fx,0,cx],[0,fy,cy],[0,0,1]]},
          "extrinsics": {"T_three": [[...4x4...]]}
        }
        """
        data = request.get_json(force=True, silent=True) or {}
        cam = data.get("camera")
        intr = data.get("intrinsics") or {}
        extr = data.get("extrinsics") or {}
        if not cam:
            return jsonify({"error": "missing 'camera'"}), 400
        if "Knew" not in intr or "width" not in intr or "height" not in intr:
            return jsonify({"error": "intrinsics must include width, height, Knew"}), 400
        if "T_three" not in extr:
            return jsonify({"error": "extrinsics must include T_three (4x4)"}), 400

        # Resolve ../calib path relative to this file
        base_dir = Path(__file__).resolve().parent
        calib_dir = (base_dir / ".." / "calib").resolve()
        calib_dir.mkdir(parents=True, exist_ok=True)
        out_path = calib_dir / f"manual_calibration_{cam}.json"

        # Write JSON file
        to_write = {
            "camera": cam,
            "intrinsics": {
                "width":  int(intr["width"]),
                "height": int(intr["height"]),
                "Knew":   intr["Knew"],
            },
            "extrinsics": {
                "T_three": extr["T_three"]
            }
        }
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(to_write, f, indent=2)
        except Exception as e:
            return jsonify({"error": f"failed to write calibration: {e}"}), 500

        # Update in-memory models so the next /api/get-state reflects it immediately
        try:
            # intrinsics
            crowd_interface._camera_models[cam] = {
                "model": "pinhole",
                "rectified": crowd_interface._camera_models.get(cam, {}).get("rectified", False),
                "width":  int(intr["width"]),
                "height": int(intr["height"]),
                "Knew":   intr["Knew"],
            }
            # extrinsics (pose)
            crowd_interface._camera_poses[f"{cam}_pose"] = extr["T_three"]
        except Exception:
            # Non-fatal; file already saved
            pass

        return jsonify({"status": "ok", "path": str(out_path)})
    
    return app