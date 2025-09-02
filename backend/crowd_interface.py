from __future__ import annotations
import os
import cv2
import numpy as np
import time
import torch
import base64
import random
import queue
import json
import traceback
import datasets

from flask import Flask, jsonify
from flask_cors import CORS
from flask import request, make_response
from pathlib import Path
from threading import Thread, Lock
from collections import deque
from math import cos, sin

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.control_utils import sanity_check_dataset_robot_compatibility, sanity_check_dataset_name

REQUIRED_RESPONSES_PER_IMPORTANT_STATE = 10
REQUIRED_RESPONSES_PER_STATE = 1

CAM_IDS = {
    "front":       18,   # change indices / paths as needed
    "left":        16,
    "right":       2,
    "perspective": 0,
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
    def __init__(self, 
                 required_responses_per_state= REQUIRED_RESPONSES_PER_STATE,
                 required_responses_per_important_state=REQUIRED_RESPONSES_PER_IMPORTANT_STATE):

        self.cams = {}
        self.latest_goal = None
        self.goal_lock = Lock()
        self._gripper_motion = 1  # Initialize gripper motion

        self.robot_is_moving = False
        self.is_async_collection = False  # True when serving states asynchronously after recording
        
        # Reset state management
        self.is_resetting = False
        self.reset_start_time = None
        self.reset_duration_s = 0
        
        # Control events for keyboard-like functionality
        self.events = None
        
        # N responses pattern
        self.required_responses_per_state = required_responses_per_state
        self.required_responses_per_important_state = required_responses_per_important_state
        
        # Episode-based state management
        self.pending_states_by_episode = {}  # episode_id -> {state_id -> {state: dict, responses_received: int, timestamp: float}}
        self.completed_states_by_episode = {}  # episode_id -> {state_id -> {responses_received: int, completion_time: float}}
        self.completed_states_buffer_by_episode = {}  # episode_id -> {state_id -> completed_state_dict} - buffer for chronological add_frame
        self.served_states_by_episode = {}  # episode_id -> {session_id -> state_id}
        self.current_serving_episode = None  # The episode currently being served to users
        self.episodes_completed = set()  # Set of episode_ids that are fully completed
        
        self.next_state_id = 0
        self.state_lock = Lock()  # Protects all episode-based state management
        self.episodes_being_completed = set()  # Track episodes currently being processed for completion

        # Auto-labeling queue and worker thread
        self.auto_label_queue = queue.Queue()
        self.auto_label_worker_thread = None
        self.auto_label_worker_running = False

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
        
        # Teardown fence
        self._shutting_down = False
        # Start the auto-labeling worker thread
        self._start_auto_label_worker()

    def begin_shutdown(self):
        """Fence off new work immediately; endpoints will early-return."""
        self._shutting_down = True
        # Flush any remaining buffered states before shutdown
        self._flush_remaining_buffered_states()
    
    def _flush_remaining_buffered_states(self):
        """Flush any states remaining in the buffer, useful during shutdown"""
        if not self.dataset:
            return
            
        for episode_id, buffered_states in self.completed_states_buffer_by_episode.items():
            if buffered_states:
                print(f"🚿 Flushing {len(buffered_states)} remaining buffered states for episode {episode_id}")
                # Sort states by state_id to ensure chronological order
                for state_id in sorted(buffered_states.keys()):
                    self.dataset.add_frame(buffered_states[state_id])
                # Save the episode
                self.dataset.save_episode()
        
        # Clear the buffer
        self.completed_states_buffer_by_episode.clear()
        print("🧹 All buffered states flushed during shutdown")
    
    def _start_auto_label_worker(self):
        """Start the dedicated auto-labeling worker thread"""
        if self.auto_label_worker_thread is not None and self.auto_label_worker_thread.is_alive():
            return
        
        self.auto_label_worker_running = True
        self.auto_label_worker_thread = Thread(target=self._auto_label_worker, daemon=True)
        self.auto_label_worker_thread.start()
        print("🧵 Started dedicated auto-labeling worker thread")
    
    def _stop_auto_label_worker(self):
        """Stop the auto-labeling worker thread (idempotent)"""
        t = getattr(self, "auto_label_worker_thread", None)
        if not t:
            return
        if not self.auto_label_worker_running and not t.is_alive():
            return
        self.auto_label_worker_running = False
        try:
            # Non-blocking; OK if sentinel already queued/consumed
            self.auto_label_queue.put_nowait(None)
        except queue.Full:
            pass
        if t.is_alive():
            t.join(timeout=2.0)
        # Clear references and drain queue for a clean future start
        self.auto_label_worker_thread = None
        try:
            with self.auto_label_queue.mutex:
                self.auto_label_queue.queue.clear()
        except Exception:
            pass
        print("🛑 Stopped auto-labeling worker thread")
    
    def _auto_label_worker(self):
        """Dedicated worker thread that processes auto-labeling tasks from the queue"""
        while self.auto_label_worker_running and not self._shutting_down:
            try:
                # Wait for a task from the queue (blocking)
                important_state_id = self.auto_label_queue.get(timeout=1.0)
                
                # None is our stop signal
                if important_state_id is None:
                    break
                
                # Process the auto-labeling task
                self._process_auto_labeling(important_state_id)
                
                # Mark task as done
                self.auto_label_queue.task_done()
                
            except queue.Empty:
                # Timeout reached, continue the loop
                continue
            except Exception as e:
                print(f"❌ Error in auto-labeling worker: {e}")
                traceback.print_exc()
    
    def _process_auto_labeling(self, important_state_id):
        """Process auto-labeling for states before the given important state ID (episode-aware)"""
        try:
            if self._shutting_down:
                return
            with self.state_lock:
                # Find which episode the important state belongs to
                target_episode_id = None
                for episode_id, episode_states in self.pending_states_by_episode.items():
                    if important_state_id in episode_states:
                        target_episode_id = episode_id
                        break
                
                if target_episode_id is None:
                    print(f"⚠️  Important state {important_state_id} not found in any episode")
                    return
                
                episode_states = self.pending_states_by_episode[target_episode_id]
                
                # Find the PREVIOUS important state to get its action as template
                template_action = None
                previous_important_states = []
                
                # Check pending states in this episode
                for state_id, state_info in episode_states.items():
                    if (state_id < important_state_id and 
                        state_info.get("important", False) and 
                        state_info.get("actions")):
                        previous_important_states.append((state_id, state_info))
                
                # Check completed states in this episode
                if target_episode_id in self.completed_states_by_episode:
                    for state_id in self.completed_states_by_episode[target_episode_id]:
                        if state_id < important_state_id:
                            # Need to find the original state info - it might be in a separate storage
                            # For now, we'll only use pending states as completed ones don't have actions stored
                            pass
                
                if previous_important_states:
                    # Use the most recent previous important state's first action as template
                    latest_important_state_id = max(previous_important_states, key=lambda x: x[0])
                    template_action = latest_important_state_id[1]["actions"][0]
                    print(f"🎯 Using template from previous important state {latest_important_state_id[0]}")
                else:
                    # No previous important state - convert FIRST state's joint positions to tensor
                    if not episode_states:
                        return
                    
                    first_state_id = min(episode_states.keys())
                    first_state = episode_states[first_state_id]
                    
                    joint_positions = first_state['state']['joint_positions']
                    gripper_action = first_state['state'].get('gripper', 0)
                    
                    # Convert to tensor format like in record_response
                    goal_positions = []
                    for joint_name in JOINT_NAMES:
                        joint_value = joint_positions.get(joint_name, 0.0)
                        if isinstance(joint_value, (list, tuple)) and len(joint_value) > 0:
                            goal_positions.append(float(joint_value[0]))
                        else:
                            goal_positions.append(float(joint_value))
                    
                    # Handle gripper like in record_response
                    goal_positions[-1] = 0.044 if gripper_action > 0 else 0.0
                    template_action = torch.tensor(goal_positions, dtype=torch.float32)
                    print(f"🎯 Using template from first state {first_state_id} joint positions")
                
                # Find all unimportant states in this episode that are earlier than the important state
                states_to_label = []
                for state_id, state_info in episode_states.items():
                    if (state_id < important_state_id and 
                        not state_info.get("important", False) and 
                        state_info["responses_received"] < self.required_responses_per_state):
                        states_to_label.append(state_id)
                
                print(f"🏷️  Auto-labeling {len(states_to_label)} unimportant states in episode {target_episode_id} before important state {important_state_id}")
                
                # Label each unimportant state with the template action
                for state_id in states_to_label:
                    state_info = episode_states[state_id]
                    
                    # Add template actions until we reach required responses
                    while state_info["responses_received"] < self.required_responses_per_state:
                        state_info["actions"].append(template_action.clone())
                        state_info["responses_received"] += 1
                    
                    # Complete the state
                    if state_info["responses_received"] >= self.required_responses_per_state:
                        # Concatenate all action responses into a 1D tensor
                        all_actions = torch.cat(state_info["actions"][:self.required_responses_per_state], dim=0)
                        
                        # Pad with inf values to match important state shape
                        missing_responses = self.required_responses_per_important_state - self.required_responses_per_state
                        action_dim = len(JOINT_NAMES)
                        padding_size = missing_responses * action_dim
                        padding = torch.full((padding_size,), float('nan'), dtype=torch.float32)
                        all_actions = torch.cat([all_actions, padding], dim=0)
                        
                        # Create completed state for dataset
                        completed_state = {
                            **state_info["observations"],
                            "action": all_actions,
                            "task": self.task if self.task else "crowdsourced_task"
                        }
                        
                        # Buffer completed state for chronological ordering
                        if self.dataset is not None:
                            if target_episode_id not in self.completed_states_buffer_by_episode:
                                self.completed_states_buffer_by_episode[target_episode_id] = {}
                            self.completed_states_buffer_by_episode[target_episode_id][state_id] = completed_state
                        
                        # Move to episode-based completed states
                        if target_episode_id not in self.completed_states_by_episode:
                            self.completed_states_by_episode[target_episode_id] = {}
                        
                        self.completed_states_by_episode[target_episode_id][state_id] = {
                            "responses_received": state_info["responses_received"],
                            "completion_time": time.time()
                        }
                        
                        print(f"🏷️  Auto-labeled and completed state {state_id} in episode {target_episode_id}")
                
                # Remove auto-labeled states from pending (episode-aware)
                for state_id in states_to_label:
                    if state_id in episode_states:
                        del episode_states[state_id]
                
                if states_to_label:
                    print(f"🗑️  Removed {len(states_to_label)} auto-labeled states from episode {target_episode_id}. Remaining: {len(episode_states)}")
                    
        except Exception as e:
            print(f"❌ Error in _process_auto_labeling: {e}")
            import traceback
            traceback.print_exc()
    
    def set_events(self, events):
        """Set the events object for keyboard-like control functionality"""
        self.events = events
    
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
        
        # Update dataset action shape to accommodate crowd responses
        self._update_dataset_action_shape()
    
    def _update_dataset_action_shape(self):
        """Update the dataset's action feature shape to include crowd responses dimension"""
        if self.dataset is not None and "action" in self.dataset.features:
            from datasets import Sequence, Value, Features
            from lerobot.common.datasets.utils import get_hf_features_from_features
            
            original_action_dim = self.dataset.features["action"]["shape"][-1]  # Get the last dimension (joint count)
            new_action_shape = (self.required_responses_per_important_state * original_action_dim,)
            
            # Update both the dataset features and metadata
            self.dataset.features["action"]["shape"] = new_action_shape
            self.dataset.meta.features["action"]["shape"] = new_action_shape
            
            # Recreate the HF dataset with updated features
            if self.dataset.hf_dataset is not None:
                # Get new HF features from the updated self.features
                new_hf_features = get_hf_features_from_features(self.dataset.features)
                
                # Create a new empty dataset with the correct features
                ft_dict = {col: [] for col in new_hf_features}
                new_hf_dataset = datasets.Dataset.from_dict(ft_dict, features=new_hf_features, split="train")
                
                # Apply the same transform
                from lerobot.common.datasets.utils import hf_transform_to_torch
                new_hf_dataset.set_transform(hf_transform_to_torch)
                
                # Replace the old dataset
                self.dataset.hf_dataset = new_hf_dataset

            print(f"📐 Updated dataset action shape to {new_action_shape} (crowd_responses={self.required_responses_per_important_state}, joints={original_action_dim})")

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
        """Close all cameras and stop workers (idempotent)"""
        # Stop auto-labeling worker (safe if already stopped)
        try:
            self._stop_auto_label_worker()
        except Exception:
            pass

        # Stop background capture workers
        if getattr(self, "_cap_running", False):
            self._cap_running = False
            for t in list(self._cap_threads.values()):
                try:
                    for _ in range(10):  # up to ~2s total
                        t.join(timeout=0.2)
                        if not t.is_alive():
                            break
                except Exception:
                    pass
            self._cap_threads.clear()

        # Clear latest buffers
        self._latest_raw.clear()
        self._latest_ts.clear()
        self._latest_proc.clear()
        self._latest_jpeg.clear()

        # Release cameras
        for cap in list(getattr(self, "cams", {}).values()):
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
    def add_state(self, 
                  joint_positions: dict, 
                  gripper_motion: int = None, 
                  obs_dict: dict[str, torch.Tensor] = None,
                  episode_id: str = None):
        # Fix: Ensure episode_id is never None to prevent None key in dictionary
        if episode_id is None:
            episode_id = "0"  # Default episode ID
            
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
        
        # Episode-based state management - fast path to minimize latency
        state_id = self.next_state_id
        self.next_state_id += 1
        
        # Deep copy obs_dict tensors (necessary for correctness)
        obs_dict_deep_copy = {}
        for key, value in obs_dict.items():
            if isinstance(value, torch.Tensor):
                obs_dict_deep_copy[key] = value.clone().detach()
            else:
                obs_dict_deep_copy[key] = value

        state_info = {
            "state": frontend_state.copy(),
            "observations": obs_dict_deep_copy,
            "actions": [],
            "responses_received": 0,
            "timestamp": time.time(),
            "served_during_stationary": None,
            "important": False,
            "episode_id": episode_id
        }
        
        # Quick episode-based assignment with minimal lock time
        with self.state_lock:
            # Initialize episode containers if needed
            if episode_id not in self.pending_states_by_episode:
                self.pending_states_by_episode[episode_id] = {}
                self.completed_states_by_episode[episode_id] = {}
                self.served_states_by_episode[episode_id] = {}
            
            # Add state to episode
            self.pending_states_by_episode[episode_id][state_id] = state_info
            
            # Set current serving episode if none set
            if self.current_serving_episode is None:
                self.current_serving_episode = episode_id
        
        print(f"🟢 State {state_id} added to episode {episode_id}. Pending: {len(self.pending_states_by_episode.get(episode_id, {}))}")

    def set_last_state_to_important(self):
        """Mark the most recent state as important (across all episodes)"""
        with self.state_lock:
            # Find the most recent state across all episodes
            latest_state_id = None
            latest_episode_id = None
            latest_timestamp = 0
            
            for episode_id, episode_states in self.pending_states_by_episode.items():
                if not episode_states:
                    continue
                    
                # Find the most recent state in this episode
                for state_id, state_info in episode_states.items():
                    if state_info["timestamp"] > latest_timestamp:
                        latest_timestamp = state_info["timestamp"]
                        latest_state_id = state_id
                        latest_episode_id = episode_id
            
            if latest_state_id is not None and latest_episode_id is not None:
                # Mark the most recent state as important
                self.pending_states_by_episode[latest_episode_id][latest_state_id]['important'] = True
                print(f"🔴 Marked state {latest_state_id} in episode {latest_episode_id} as important")
                
                # Trigger auto-labeling for states in the same episode
                self.auto_label_previous_states(latest_state_id)
            else:
                print("⚠️  No pending states found to mark as important")

    def auto_label_previous_states(self, important_state_id):
        """
        Queue an auto-labeling task for unimportant pending states that are 
        earlier than the given important state ID.
        """
        try:
            if self._shutting_down:
                return
            # Add the task to the queue (non-blocking)
            self.auto_label_queue.put_nowait(important_state_id)
            print(f"📝 Queued auto-labeling task for states before {important_state_id}")
        except queue.Full:
            print(f"⚠️  Auto-labeling queue is full, skipping task for state {important_state_id}")
        except Exception as e:
            print(f"❌ Error queuing auto-labeling task: {e}")
    
    def get_latest_state(self, session_id: str = "default") -> dict:
        """
        Get a state that needs responses from the current serving episode.
        Episodes are served one by one - only move to next episode after current is complete.
        """
        if self._shutting_down:
            return {}
        with self.state_lock:
            # Get current serving episode or find the next one
            if (self.current_serving_episode is None or 
                self.current_serving_episode in self.episodes_completed or
                not self.pending_states_by_episode.get(self.current_serving_episode)):
                
                # Find next episode to serve
                available_episodes = [ep_id for ep_id in self.pending_states_by_episode.keys() 
                                    if ep_id not in self.episodes_completed and 
                                    self.pending_states_by_episode[ep_id] and
                                    ep_id is not None]  # Exclude None keys
                
                if not available_episodes:
                    return {}  # No episodes to serve
                
                # Move to the earliest available episode (safe since None is filtered out)
                self.current_serving_episode = min(available_episodes)
                print(f"🎬 Now serving episode {self.current_serving_episode}")
            
            episode_states = self.pending_states_by_episode[self.current_serving_episode]
            if not episode_states:
                return {}
            
            # Select state based on robot movement status and async collection mode
            if self.robot_is_moving is False and not self.is_async_collection:
                # Robot not moving and not in async mode - prioritize LATEST state from ALL EPISODES for immediate execution
                latest_state_id = None
                latest_timestamp = 0
                latest_episode = None
                
                # Find the most recent state across all episodes
                for ep_id, ep_states in self.pending_states_by_episode.items():
                    if not ep_states:
                        continue
                    for state_id, state_data in ep_states.items():
                        if state_data["timestamp"] > latest_timestamp:
                            latest_timestamp = state_data["timestamp"]
                            latest_state_id = state_id
                            latest_episode = ep_id
                
                if latest_state_id is not None:
                    state_info = self.pending_states_by_episode[latest_episode][latest_state_id]
                    episode_states = self.pending_states_by_episode[latest_episode]
                else:
                    # Fallback if no states found
                    return {}
            else:
                # Robot moving OR async collection mode - select random state for diverse data collection from current serving episode
                random_state_id = random.choice(list(episode_states.keys()))
                state_info = episode_states[random_state_id]
                latest_state_id = random_state_id
            
            # Mark state as served
            if state_info["served_during_stationary"] is None:
                state_info["served_during_stationary"] = not (self.robot_is_moving or self.is_async_collection)
            
            # Track which state was served to this session (episode-aware)
            serving_episode = latest_episode if (self.robot_is_moving is False and not self.is_async_collection) else self.current_serving_episode
            if serving_episode not in self.served_states_by_episode:
                self.served_states_by_episode[serving_episode] = {}
            self.served_states_by_episode[serving_episode][session_id] = latest_state_id
            
            # Prepare response
            state = state_info["state"].copy()
            state["state_id"] = latest_state_id
            state["episode_id"] = serving_episode
            
            if self.is_async_collection:
                status = "async_collection"
            elif self.robot_is_moving:
                status = "moving"
            else:
                status = "stationary"
            print(f"🎯 Serving state {latest_state_id} from episode {serving_episode} to session {session_id} ({status})")
            return state
    
    def _is_submission_meaningful(self, submitted_joints: dict, original_joints: dict, submitted_gripper: int = None, original_gripper: int = None, threshold: float = 0.01) -> bool:
        """
        Check if the submitted joint positions or gripper action are meaningfully different from the original state.
        Now allows small movements and no-change actions as they can be meaningful (e.g., "stay in place", "fine adjustments").
        
        Args:
            submitted_joints: Joint positions from user submission
            original_joints: Original joint positions from the state
            submitted_gripper: Gripper action from user submission (optional)
            original_gripper: Original gripper action from the state (optional)
            threshold: Minimum difference threshold (in radians/meters) - now only used for logging
        
        Returns:
            True (always accepts submissions now - small/no movements are valid actions)
        """
        # Check joint positions for logging purposes
        total_diff = 0.0
        joint_count = 0
        
        for joint_name in JOINT_NAMES[:-1]:  # Exclude gripper for now
            if joint_name in submitted_joints and joint_name in original_joints:
                # Get the first element if it's a list/array, otherwise use directly
                submitted_val = submitted_joints[joint_name]
                if isinstance(submitted_val, (list, tuple)):
                    submitted_val = submitted_val[0]
                
                original_val = original_joints[joint_name]
                if isinstance(original_val, (list, tuple)):
                    original_val = original_val[0]
                
                diff = abs(float(submitted_val) - float(original_val))
                total_diff += diff
                joint_count += 1
        
        # Check if gripper action has changed (for logging)
        gripper_changed = False
        if submitted_gripper is not None and original_gripper is not None:
            # Convert to standardized values (1 for open, -1 for close)
            submitted_gripper_norm = 1 if submitted_gripper > 0 else -1
            original_gripper_norm = 1 if original_gripper > 0 else -1
            gripper_changed = submitted_gripper_norm != original_gripper_norm
        
        avg_diff = total_diff / max(joint_count, 1)
        movement_type = "no_change" if avg_diff < 0.001 else ("small_movement" if avg_diff < threshold else "large_movement")
        
        print(f"📏 Submission accepted: joints_diff={avg_diff:.4f}, gripper_changed={gripper_changed}, movement_type={movement_type}")
        
        # Always return True - all submissions are now considered meaningful
        # Small movements and no-change actions are valid responses in crowdsourcing
        return True

    def record_response(self, response_data: dict, session_id: str = "default") -> bool:
        """
        Record a response for a specific state in the episode-based system.
        Returns True if this completes the required responses for a state.
        """
        if self._shutting_down:
            print("⚠️  Ignoring submission during shutdown")
            return False
        with self.state_lock:
            # Get state_id and episode_id from response
            state_id = response_data.get("state_id")
            episode_id = response_data.get("episode_id")
            
            if state_id is None:
                print("⚠️  No state_id in response data")
                return False
            
            # Find the state in episode-based storage
            state_info = None
            found_episode = episode_id
            
            if episode_id is not None and episode_id in self.pending_states_by_episode:
                state_info = self.pending_states_by_episode[episode_id].get(state_id)
                found_episode = episode_id
            
            if not state_info:
                print(f"⚠️  State {state_id} not found in pending states")
                return False
            
            required_responses = (self.required_responses_per_important_state 
                                if state_info['important'] else self.required_responses_per_state)
            
            # Extract joint positions and gripper action from response
            joint_positions = response_data.get("joint_positions", {})
            gripper_action = response_data.get("gripper", 0)
            
            # Check meaningfulness (always returns True but provides logging)
            if not response_data.get("task_already_completed", False):
                original_joints = state_info["state"].get("joint_positions", {})
                original_gripper = state_info["state"].get("gripper", 0)
                self._is_submission_meaningful(joint_positions, original_joints, gripper_action, original_gripper)
            
            # Set as goal if first response to stationary-served state
            if (state_info["responses_received"] == 0 and 
                state_info["served_during_stationary"] is True):
                self.latest_goal = response_data
                print(f"🎯 Setting goal for immediate execution - first response to stationary state {state_id}")
            
            # Record the response
            state_info["responses_received"] += 1
            
            # Convert response to tensor format
            goal_positions = []
            for joint_name in JOINT_NAMES:
                joint_value = joint_positions.get(joint_name, 0.0)
                if isinstance(joint_value, (list, tuple)) and len(joint_value) > 0:
                    goal_positions.append(float(joint_value[0]))
                else:
                    goal_positions.append(float(joint_value))
            
            goal_positions[-1] = 0.044 if gripper_action > 0 else 0.0
            goal_positions = torch.tensor(goal_positions, dtype=torch.float32)
            state_info["actions"].append(goal_positions)

            print(f"🔔 Response recorded for state {state_id} in episode {found_episode} ({state_info['responses_received']}/{required_responses})")
            
            # Check if state is complete
            if state_info["responses_received"] >= required_responses:
                # Complete the state and add to dataset
                all_actions = torch.cat(state_info["actions"][:required_responses], dim=0)

                if required_responses < self.required_responses_per_important_state:
                    # Pad with inf values
                    missing_responses = self.required_responses_per_important_state - required_responses
                    action_dim = len(JOINT_NAMES)
                    padding_size = missing_responses * action_dim
                    padding = torch.full((padding_size,), float('nan'), dtype=torch.float32)
                    all_actions = torch.cat([all_actions, padding], dim=0)

                # Create complete frame for dataset
                completed_state = {
                    **state_info["observations"],
                    "action": all_actions,
                    "task": self.task if self.task else "crowdsourced_task"
                }
                
                # Buffer completed state for chronological ordering
                if found_episode not in self.completed_states_buffer_by_episode:
                    self.completed_states_buffer_by_episode[found_episode] = {}
                self.completed_states_buffer_by_episode[found_episode][state_id] = completed_state
                
                # Move to completed states
                if found_episode not in self.completed_states_by_episode:
                    self.completed_states_by_episode[found_episode] = {}
                self.completed_states_by_episode[found_episode][state_id] = {
                    "responses_received": state_info["responses_received"],
                    "completion_time": time.time(),
                    "is_important": state_info.get("important", False)
                }
                
                # Remove from pending
                del self.pending_states_by_episode[found_episode][state_id]
                
                # Check if episode is complete
                if not self.pending_states_by_episode[found_episode]:
                    # Mark episode as being completed to prevent race conditions with is_recording()
                    self.episodes_being_completed.add(found_episode)
                    print(f"🎬 Episode {found_episode} completed! Saving episode to dataset...")
                    
                    try:
                        # Add buffered states to dataset in chronological order
                        if found_episode in self.completed_states_buffer_by_episode:
                            buffered_states = self.completed_states_buffer_by_episode[found_episode]
                            # Sort states by state_id to ensure chronological order
                            for state_id in sorted(buffered_states.keys()):
                                self.dataset.add_frame(buffered_states[state_id])
                            print(f"📚 Added {len(buffered_states)} states to dataset in chronological order")
                            # Clean up buffer
                            del self.completed_states_buffer_by_episode[found_episode]
                        
                        self.episodes_completed.add(found_episode)
                        self.dataset.save_episode()
                        
                        # Clean up episode data structures  
                        del self.pending_states_by_episode[found_episode]
                        if found_episode in self.served_states_by_episode:
                            del self.served_states_by_episode[found_episode]

                        # NEW: drop completed episode from monitor memory so it won't be transferred to the frontend
                        # This removes the (potentially large) completed-state bookkeeping for this episode.
                        if found_episode in self.completed_states_by_episode:
                            del self.completed_states_by_episode[found_episode]
                            print(f"🧹 Dropped completed episode {found_episode} from monitor memory")
                        
                        # Update current_serving_episode if this was the current one
                        if self.current_serving_episode == found_episode:
                            # Find next episode to serve
                            available_episodes = [ep_id for ep_id in self.pending_states_by_episode.keys() 
                                                if ep_id not in self.episodes_completed and 
                                                self.pending_states_by_episode[ep_id] and
                                                ep_id is not None]
                            
                            if available_episodes:
                                self.current_serving_episode = min(available_episodes)
                                print(f"🎬 Now serving next episode {self.current_serving_episode}")
                            else:
                                self.current_serving_episode = None
                                print(f"🏁 All episodes completed, no more episodes to serve")
                    finally:
                        # Always remove from being_completed set, even if there's an error
                        self.episodes_being_completed.discard(found_episode)
                
                print(f"✅ State {state_id} completed in episode {found_episode}")
                return True
            
            return False
    
    def get_pending_states_info(self) -> dict:
        """Get episode-based state information for monitoring"""
        with self.state_lock:
            episodes_info = {}
            total_pending = 0
            
            # Only expose episodes that still have pending states (hide completed episodes entirely)
            all_episode_ids = set(self.pending_states_by_episode.keys())
            
            # Process each episode
            for episode_id in sorted(all_episode_ids):
                episode_states = {}
                
                # Add pending states from this episode
                if episode_id in self.pending_states_by_episode:
                    for state_id, info in self.pending_states_by_episode[episode_id].items():
                        required_responses = (self.required_responses_per_important_state 
                                            if info['important'] else self.required_responses_per_state)
                        episode_states[state_id] = {
                            "responses_received": info["responses_received"],
                            "responses_needed": required_responses - info["responses_received"],
                            "is_important": info.get("important", False)
                        }
                        total_pending += 1
                
                # Add completed states from this episode
                if episode_id in self.completed_states_by_episode:
                    for state_id, info in self.completed_states_by_episode[episode_id].items():
                        episode_states[state_id] = {
                            "responses_received": info["responses_received"],
                            "responses_needed": 0,  # Completed
                            "is_important": info.get("is_important", False)
                        }
                
                episodes_info[episode_id] = {
                    "states": episode_states,
                    "pending_count": len(self.pending_states_by_episode.get(episode_id, {})),
                    "completed_count": len(self.completed_states_by_episode.get(episode_id, {})),
                    "is_current_serving": episode_id == self.current_serving_episode,
                    "is_completed": episode_id in self.episodes_completed
                }
            
            return {
                "total_pending": total_pending,
                "current_serving_episode": self.current_serving_episode,
                "required_responses_per_state": self.required_responses_per_state,
                "required_responses_per_important_state": self.required_responses_per_important_state,
                "episodes": episodes_info
            }
    
    # --- Goal Management ---
    def submit_goal(self, goal_data: dict):
        """Submit a new goal from the frontend - goal setting now handled in record_response"""
        # Goal setting logic has been moved to record_response method
        # This method is kept for API compatibility but does nothing
        pass
    
    def get_latest_goal(self) -> dict | None:
        """Get and clear the latest goal (for robot loop to consume)"""
        goal = self.latest_goal
        self.latest_goal = None
        return goal
    
    def has_pending_goal(self) -> bool:
        """Check if there's a pending goal"""
        return self.latest_goal is not None
    
    # --- Robot Movement State Management ---
    def set_robot_moving(self, is_moving: bool = True):
        """Set the robot movement state"""
        self.robot_is_moving = is_moving
        if is_moving:
            self.is_async_collection = False  # Clear async mode when robot is actually moving
        status = "MOVING" if is_moving else "STATIONARY"
        emoji = "🏃" if is_moving else "🛑"
        print(f"{emoji} Robot state set to: {status}")
    
    def is_robot_moving(self) -> bool:
        """Get current robot movement state"""
        return self.robot_is_moving
    
    def set_async_collection(self, is_async: bool = True):
        """Set asynchronous data collection mode (serving random states after recording)"""
        self.is_async_collection = is_async
        if is_async:
            self.robot_is_moving = False  # Robot is not actually moving during async collection
        status = "ASYNC COLLECTION" if is_async else "NORMAL"
        emoji = "🔄" if is_async else "⏸️"
        print(f"{emoji} Data collection mode set to: {status}")
    
    def is_async_collection_mode(self) -> bool:
        """Check if in asynchronous data collection mode"""
        return self.is_async_collection
    
    # --- Reset State Management ---
    def start_reset(self, duration_s: float):
        """Start the reset countdown timer"""
        self.is_resetting = True
        self.reset_start_time = time.time()
        self.reset_duration_s = duration_s
        print(f"🔄 Starting reset countdown: {duration_s}s")
    
    def stop_reset(self):
        """Stop the reset countdown timer"""
        self.is_resetting = False
        self.reset_start_time = None
        self.reset_duration_s = 0
        print(f"✅ Reset completed")
    
    def get_reset_countdown(self) -> float:
        """Get remaining reset time in seconds, or 0 if not resetting"""
        if not self.is_resetting or self.reset_start_time is None:
            return 0
        
        elapsed = time.time() - self.reset_start_time
        remaining = max(0, self.reset_duration_s - elapsed)
        
        # Auto-stop when countdown reaches 0
        if remaining <= 0 and self.is_resetting:
            self.stop_reset()
        
        return remaining
    
    def is_in_reset(self) -> bool:
        """Check if currently in reset state"""
        return self.is_resetting and self.get_reset_countdown() > 0
    
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
        """Check if there are any pending states across all episodes or episodes being completed"""
        with self.state_lock:
            # Check for pending states or episodes currently being completed
            has_pending_states = any(len(episode_states) > 0 for episode_states in self.pending_states_by_episode.values())
            has_episodes_being_completed = len(self.episodes_being_completed) > 0
            
            is_recording = has_pending_states or has_episodes_being_completed
            if has_episodes_being_completed:
                print(f"🔄 Recording status: {is_recording} (pending_states: {has_pending_states}, episodes_being_completed: {list(self.episodes_being_completed)})")
            
            return is_recording

def create_flask_app(crowd_interface: CrowdInterface) -> Flask:
    """Create and configure Flask app with the crowd interface"""
    app = Flask(__name__)
    CORS(app, origins=["*"], 
         allow_headers=["Content-Type", "ngrok-skip-browser-warning"],
         methods=["GET", "POST", "OPTIONS"])
    
    # Add ngrok-specific headers
    @app.after_request
    def after_request(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,ngrok-skip-browser-warning'
        response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
        return response
    
    # Handle preflight OPTIONS requests
    @app.before_request
    def handle_preflight():
        if request.method == "OPTIONS":
            response = make_response()
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add('Access-Control-Allow-Headers', "Content-Type,ngrok-skip-browser-warning")
            response.headers.add('Access-Control-Allow-Methods', "GET,POST,OPTIONS")
            return response
    
    @app.route("/api/get-state")
    def get_state():
        if crowd_interface._shutting_down:
            return jsonify({}), 200
        current_time = time.time()
        
        # Generate or retrieve session ID from request headers or IP
        session_id = request.headers.get('X-Session-ID', request.remote_addr)
        
        state = crowd_interface.get_latest_state(session_id)
        # print(f"🔍 Flask route /api/get-state called at {current_time}")
        # print(f"🔍 Pending states: {len(crowd_interface.pending_states)}")
        payload = crowd_interface._state_to_json(state)
        
        # Add hardcoded prompt text
        payload["prompt"] = f"Task: {crowd_interface.task} What should the arm do next?"

        response = jsonify(payload)
        # Prevent caching
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    
    @app.route("/api/test")
    def test():
        # Count total states across all episodes
        total_states = sum(len(states) for states in crowd_interface.pending_states_by_episode.values())
        return jsonify({"message": "Flask server is working", "states_count": total_states})
    
    @app.route("/api/submit-goal", methods=["POST"])
    def submit_goal():
        if crowd_interface._shutting_down:
            return jsonify({"status": "shutting_down"}), 503
        
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
    
    @app.route("/api/monitor/latest-state", methods=["GET"])
    def monitor_latest_state():
        """
        Read-only monitoring endpoint for episode-based state monitoring
        """
        try:
            with crowd_interface.state_lock:
                # Check if we have any pending states across all episodes
                all_pending = {}
                current_episode = crowd_interface.current_serving_episode
                
                for episode_id, episode_states in crowd_interface.pending_states_by_episode.items():
                    all_pending.update(episode_states)
                
                if not all_pending:
                    # No pending states - check if we have any completed states to show info about
                    total_completed_states = sum(len(states) for states in crowd_interface.completed_states_by_episode.values())
                    completed_episodes = list(crowd_interface.episodes_completed)
                    
                    return jsonify({
                        "status": "no_pending_states",
                        "message": "No pending states.",
                        "views": {},
                        "total_pending_states": 0,
                        "current_serving_episode": current_episode,
                        "robot_moving": crowd_interface.is_robot_moving(),
                        "is_async_collection": crowd_interface.is_async_collection_mode(),
                        "is_resetting": crowd_interface.is_in_reset(),
                        "reset_countdown": crowd_interface.get_reset_countdown(),
                        "timestamp": time.time()
                    })
                
                # Get the newest state across all episodes
                newest_state_id = max(all_pending.keys())
                newest_state_data = all_pending[newest_state_id]
                newest_state = newest_state_data["state"]
                episode_id = newest_state_data["episode_id"]
            
            # Build monitoring response with episode information
            monitoring_data = {
                "status": "success",
                "state_id": newest_state_id,
                "episode_id": episode_id,
                "current_serving_episode": current_episode,
                "timestamp": newest_state_data["timestamp"],
                "responses_received": newest_state_data["responses_received"],
                "responses_required": (crowd_interface.required_responses_per_important_state 
                                     if newest_state_data.get("important", False) else crowd_interface.required_responses_per_state),
                "is_important": newest_state_data.get("important", False),
                "views": newest_state.get("views", {}),
                "joint_positions": newest_state.get("joint_positions", {}),
                "gripper": newest_state.get("gripper", 0),
                "robot_moving": crowd_interface.is_robot_moving(),
                "is_async_collection": crowd_interface.is_async_collection_mode(),
                "is_resetting": crowd_interface.is_in_reset(),
                "reset_countdown": crowd_interface.get_reset_countdown(),
                "total_pending_states": len(all_pending),
                "current_time": time.time()
            }
            
            response = jsonify(monitoring_data)
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response
            
        except Exception as e:
            # Don't let monitoring errors affect the main system
            return jsonify({
                "status": "error",
                "message": f"Monitoring error: {str(e)}",
                "timestamp": time.time()
            }), 500
    
    @app.route("/api/control/next-episode", methods=["POST"])
    def next_episode():
        """Trigger next episode (equivalent to 'q' keyboard input)"""
        try:
            if crowd_interface.events is not None:
                print("API trigger: Exiting current loop...")
                crowd_interface.events["exit_early"] = True
                return jsonify({"status": "success", "message": "Next episode triggered"})
            else:
                return jsonify({"status": "error", "message": "Events not initialized"}), 400
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route("/api/control/rerecord", methods=["POST"])
    def rerecord_episode():
        """Trigger re-record episode (equivalent to 'r' keyboard input)"""
        try:
            if crowd_interface.events is not None:
                print("API trigger: Exiting loop and re-record the last episode...")
                crowd_interface.events["rerecord_episode"] = True
                crowd_interface.events["exit_early"] = True
                return jsonify({"status": "success", "message": "Re-record episode triggered"})
            else:
                return jsonify({"status": "error", "message": "Events not initialized"}), 400
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route("/api/control/stop", methods=["POST"])
    def stop_recording():
        """Trigger stop recording (equivalent to 'x' keyboard input)"""
        try:
            if crowd_interface.events is not None:
                print("API trigger: Stopping data recording...")
                crowd_interface.events["stop_recording"] = True
                crowd_interface.events["exit_early"] = True
                return jsonify({"status": "success", "message": "Stop recording triggered"})
            else:
                return jsonify({"status": "error", "message": "Events not initialized"}), 400
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route("/api/control/start-episode", methods=["POST"])
    def start_episode():
        """Skip remaining reset time and start the next episode immediately"""
        try:
            if crowd_interface.is_in_reset():
                crowd_interface.stop_reset()
                return jsonify({"status": "success", "message": "Reset skipped, starting episode"})
            else:
                return jsonify({"status": "error", "message": "Not currently in reset state"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
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
    
    @app.route("/api/task-already-completed", methods=["POST"])
    def task_already_completed():
        """Handle 'Task Already Completed' submissions by recording original state as goal"""
        if crowd_interface._shutting_down:
            return jsonify({"status": "shutting_down"}), 503
        try:
            # Try to get JSON data with force=True to bypass Content-Type check
            try:
                data = request.get_json(force=True) or {}
            except Exception:
                # Fallback to manual JSON parsing if that fails
                try:
                    data = json.loads(request.get_data(as_text=True)) if request.get_data() else {}
                except Exception:
                    data = {}
            
            session_id = request.headers.get("X-Session-ID", "default")
            
            # Get the state_id and episode_id from request
            state_id = data.get("state_id")
            episode_id = data.get("episode_id")
            if state_id is None:
                return jsonify({"error": "state_id is required"}), 400
            
            # Get the original state to use its joint positions as the goal
            with crowd_interface.state_lock:
                # Use episode_id if provided, otherwise fall back to session tracking
                original_state = None
                found_episode = None
                
                if episode_id and episode_id in crowd_interface.pending_states_by_episode:
                    if state_id in crowd_interface.pending_states_by_episode[episode_id]:
                        original_state = crowd_interface.pending_states_by_episode[episode_id][state_id]["state"]
                        found_episode = episode_id
                elif not episode_id:
                    # Use session tracking to find the correct episode
                    for ep_id, session_states in crowd_interface.served_states_by_episode.items():
                        if session_id in session_states and session_states[session_id] == state_id:
                            if ep_id in crowd_interface.pending_states_by_episode and state_id in crowd_interface.pending_states_by_episode[ep_id]:
                                original_state = crowd_interface.pending_states_by_episode[ep_id][state_id]["state"]
                                found_episode = ep_id
                                break
                
                if original_state is None:
                    return jsonify({"error": f"State {state_id} not found or already completed"}), 404
                
                original_joints = original_state.get("joint_positions", {}).copy()
                for joint_name in original_joints.keys():
                    original_joints[joint_name] = [original_joints[joint_name]]
                original_gripper = original_state.get("gripper", 0)
            
            # Create response data using original joint positions as the goal
            # This is equivalent to the user clicking "Confirm" without moving any sliders
            response_data = {
                "state_id": state_id,
                "episode_id": found_episode,             # Include the found episode_id
                "joint_positions": original_joints,  # Use original positions as goal
                "gripper": original_gripper,          # Use original gripper state as goal
                "task_already_completed": True        # Flag to indicate this was a "no change needed" submission
            }
            
            # Reuse existing response recording infrastructure
            completion_status = crowd_interface.record_response(response_data, session_id)
            
            if completion_status:
                return jsonify({"status": "success", "message": "Task already completed response recorded and state completed"})
            else:
                return jsonify({"status": "success", "message": "Task already completed response recorded"})
                
        except Exception as e:
            print(f"❌ Error in task-already-completed: {e}")
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500
    
    return app