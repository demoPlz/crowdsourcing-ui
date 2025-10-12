import os
import cv2
import tempfile
import numpy as np
import time
import torch
import base64
import random
import queue
import json

import datasets
import re
import mimetypes
from pathlib import Path
from threading import Thread, Lock, Timer
from math import cos, sin

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.control_utils import sanity_check_dataset_robot_compatibility, sanity_check_dataset_name

REQUIRED_RESPONSES_PER_IMPORTANT_STATE = 10
REQUIRED_RESPONSES_PER_STATE = 1

CAM_IDS = {
    "front":       18,   # change indices / paths as needed
    "left":        4,
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
                 required_responses_per_critical_state=REQUIRED_RESPONSES_PER_IMPORTANT_STATE,
                 autofill_critical_states: bool = False,
                 num_autofill_actions: int | None = None,
                 use_manual_prompt: bool = False,
                 # --- saving critical-state cam_main frames ---
                 save_maincam_sequence: bool = False,
                 prompt_sequence_dir: str | None = None,
                 prompt_sequence_clear: bool = False,
                 # used ONLY for prompt substitution and demo assets
                 prompt_task_name: str | None = None,
                 # --- demo video recording ---
                 record_demo_videos: bool = False,
                 demo_videos_dir: str | None = None,
                 demo_videos_clear: bool = False,
                 # --- read-only demo video display (independent of recording) ---
                 show_demo_videos: bool = False,
                 show_videos_dir: str | None = None,
    ):
        
        # --- UI prompt mode (simple vs VLM vs MANUAL) ---
        self.use_manual_prompt = bool(use_manual_prompt or int(os.getenv("USE_MANUAL_PROMPT", "0")))

        # -------- Observation disk cache (spills heavy per-state obs to disk) --------
        # Set CROWD_OBS_CACHE to override where temporary per-state observations are stored.
        self._obs_cache_root = Path(os.getenv("CROWD_OBS_CACHE", os.path.join(tempfile.gettempdir(), "crowd_obs_cache")))
        try:
            self._obs_cache_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        self.cams = {}
        self.latest_goal = None
        self.goal_lock = Lock()
        self._gripper_motion = 1  # Initialize gripper motion
        
        # Reset state management
        self.is_resetting = False
        self.reset_start_time = None
        self.reset_duration_s = 0
        
        # Control events for keyboard-like functionality
        self.events = None
        
        # N responses pattern
        self.required_responses_per_state = required_responses_per_state
        self.required_responses_per_critical_state = required_responses_per_critical_state
        self.autofill_critical_states = bool(autofill_critical_states)
        # If not specified, default to "complete on first submission"
        if num_autofill_actions is None:
            self.num_autofill_actions = self.required_responses_per_critical_state
        else:
            self.num_autofill_actions = int(num_autofill_actions)
        # Clamp to [1, required_responses_per_critical_state]
        self.num_autofill_actions = max(1, min(self.num_autofill_actions,
                                               self.required_responses_per_critical_state))
        
        # Episode-based state management
        self.pending_states_by_episode = {}  # episode_id -> {state_id -> {state: dict, responses_received: int}}
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
        # Task used for UI fallback and dataset frames → always cfg.single_task (set in init_dataset)
        self.task_text = None
        # Task name used for prompt placeholder substitution and demo images (from --task-name)
        self.prompt_task_name = (prompt_task_name or None)

        # Background capture state
        self._cap_threads: dict[str, Thread] = {}
        self._cap_running: bool = False
        self._latest_jpeg: dict[str, str] = {}
        # JPEG quality for base64 encoding (override with env JPEG_QUALITY)
        self._jpeg_quality = int(os.getenv("JPEG_QUALITY", "80"))

        # Observation camera (obs_dict) live previews → background-encoded JPEGs
        self._latest_obs_jpeg: dict[str, str] = {}
        self._obs_img_queue: queue.Queue = queue.Queue(maxsize=int(os.getenv("OBS_STREAM_QUEUE", "8")))
        self._obs_img_running: bool = False
        self._obs_img_thread: Thread | None = None

        # Will be filled by _load_calibrations()
        self._undistort_maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._camera_models: dict[str, dict] = {}
        self._camera_poses = self._load_calibrations()
        # ---- Gripper tip calibration (left/right {x,y,z}) ----
        self._calib_lock = Lock()
        self._gripper_tip_calib = self._load_gripper_tip_calibration()  # {"left":{x,y,z},"right":{x,y,z}}

        # Debounced episode finalization
        self.episode_finalize_grace_s = 2.0
        self._episode_finalize_timers: dict[str, Timer] = {}


        # Precompute immutable views and camera poses to avoid per-tick allocations
        
        # Start the auto-labeling worker thread
        self._start_auto_label_worker()

        self._exec_gate_by_session: dict[str, dict] = {}

        self._active_episode_id = None
        self._start_obs_stream_worker()

        # --- Important-state cam_main image sequence sink ---
        self.save_maincam_sequence = bool(save_maincam_sequence)
        self._prompt_seq_dir = Path(prompt_sequence_dir or "prompts/drawer/snapshots").resolve()
        self._prompt_seq_index = 1
        # Track which states have been saved to maintain chronological ordering
        self._saved_sequence_states: set[tuple[str, int]] = set()  # (episode_id, state_id)
        self._max_saved_state_id: int | None = None
        # If a sequence dir was set but no prompt_task_name, infer prompt task from the leaf folder
        if (self.prompt_task_name is None) and prompt_sequence_dir:
            try:
                self.prompt_task_name = Path(prompt_sequence_dir).name
            except Exception:
                pass
        if self.save_maincam_sequence:
            try:
                self._prompt_seq_dir.mkdir(parents=True, exist_ok=True)
                if prompt_sequence_clear:
                    removed = 0
                    for p in self._prompt_seq_dir.glob("*"):
                        try:
                            if p.is_file():
                                p.unlink()
                                removed += 1
                        except Exception:
                            pass
                    print(f"🧹 Cleared {removed} files in {self._prompt_seq_dir}")
                    # Reset tracking when clearing
                    self._saved_sequence_states.clear()
                    self._max_saved_state_id = None
                self._prompt_seq_index = self._compute_next_prompt_seq_index()
                print(f"📸 Important-state capture → {self._prompt_seq_dir} (next index {self._prompt_seq_index:06d})")
            except Exception as e:
                print(f"⚠️ Could not prepare sequence directory '{self._prompt_seq_dir}': {e}")

        # --- NEW: Demo video recording ---
        self.record_demo_videos = bool(record_demo_videos)
        self._demo_videos_dir = None
        self._video_index_lock = Lock()
        self._video_index = 1   # 1-based, reset on each process start
        if self.record_demo_videos:
            if demo_videos_dir:
                self._demo_videos_dir = Path(demo_videos_dir).resolve()
            else:
                # Default: prompts/demos/{task-name}/videos
                task_name = self.prompt_task_name or "default"
                repo_root = Path(__file__).resolve().parent / ".."
                self._demo_videos_dir = (repo_root / "prompts" / task_name / "videos").resolve()
            
            try:
                self._demo_videos_dir.mkdir(parents=True, exist_ok=True)
                print(f"🎥 Demo video recording → {self._demo_videos_dir}")
                # Clear dir (recommended) so numbering restarts at 1 every run
                if demo_videos_clear:
                    removed = 0
                    for p in self._demo_videos_dir.iterdir():
                        try:
                            p.unlink()
                            removed += 1
                        except Exception:
                            pass
                    if removed:
                        print(f"🧹 Cleared {removed} old files in {self._demo_videos_dir}")
                # Ensure index starts at 1 (or next if directory not empty)
                self._video_index = self._compute_next_video_index()
            except Exception as e:
                print(f"⚠️ Could not prepare demo videos directory '{self._demo_videos_dir}': {e}")
                self.record_demo_videos = False

        # --- NEW: Demo video *display* (read-only, independent of recording) ---
        self.show_demo_videos = bool(show_demo_videos or int(os.getenv("SHOW_DEMO_VIDEOS", "0")))
        self._show_videos_dir = None
        self._show_video_exts = (".webm",)  # VP9-only

        if self.show_demo_videos:
            if show_videos_dir:
                self._show_videos_dir = Path(show_videos_dir).resolve()
            else:
                task_name = self.prompt_task_name or "default"
                repo_root = Path(__file__).resolve().parent / ".."
                self._show_videos_dir = (repo_root / "prompts" / task_name / "videos").resolve()

            try:
                self._show_videos_dir.mkdir(parents=True, exist_ok=True)
                print(f"🎬 Demo video display (read-only, VP9/WebM only) → {self._show_videos_dir}")
            except Exception as e:
                print(f"⚠️ Could not prepare show videos directory '{self._show_videos_dir}': {e}")
                self.show_demo_videos = False

        # --- Episode save behavior: datasets are always auto-saved after finalization ---
        # Manual save is only used for demo video recording workflow
        self._episodes_pending_save: set[str] = set()

    # ---------- Demo video config helpers (tell the frontend where to save) ----------
    def _repo_root(self) -> Path:
        """Root of the repo (backend assumes this file lives under <repo>/scripts or similar)."""
        return (Path(__file__).resolve().parent / "..").resolve()

    def rel_path_from_repo(self, p: str | Path | None) -> str | None:
        if not p:
            return None
        try:
            rp = Path(p).resolve()
            return str(rp.relative_to(self._repo_root()))
        except Exception:
            # If not inside the repo root, return the basename as a safe hint.
            return os.path.basename(str(p))

    # ---------- Prompt-mode helpers (manual or VLM) ----------
    def set_prompt_ready(self, state_info: dict, episode_id: int, state_id: int, text: str | None, video_id: int | None) -> None:
        """Set text/video prompt fields and mark as ready."""
        state_info["text_prompt"] = text  # Updated field name
        state_info["video_prompt"] = video_id  # Updated field name
        state_info["prompt_ready"] = True
        
        # Check if this is a critical state with "end." text - auto-fill with current position
        if text and text.strip().lower() == "end.":
            self._auto_fill_end_state_locked(state_info, episode_id, state_id)

    def _auto_fill_end_state_locked(self, state_info: dict, episode_id: int, state_id: int) -> None:
        """
        Auto-fill an critical state labeled as "end." with multiple copies of its current position.
        MUST be called with self.state_lock already held.
        """
        # Direct access to joint positions and gripper in flattened structure
        joint_positions = state_info.get('joint_positions', {})
        gripper_action = state_info.get('gripper', 0)
        
        # Convert joint positions to action tensor (same as autolabel logic)
        goal_positions = []
        for joint_name in JOINT_NAMES:
            v = joint_positions.get(joint_name, 0.0)
            v = float(v[0]) if isinstance(v, (list, tuple)) and len(v) > 0 else float(v)
            goal_positions.append(v)
        # Set gripper position based on gripper action
        goal_positions[-1] = 0.044 if gripper_action > 0 else 0.0
        
        position_action = torch.tensor(goal_positions, dtype=torch.float32)
        
        state_info['actions'] = [position_action for _ in range(self.required_responses_per_critical_state)]
        all_actions = torch.cat(state_info["actions"][:self.required_responses_per_critical_state], dim=0)

        state_info['action_to_save'] = all_actions

        self.completed_states_buffer_by_episode[episode_id][state_id] = state_info
        self.completed_states_by_episode[episode_id][state_id] = state_info

        del self.pending_states_by_episode[episode_id][state_id]

        if not self.pending_states_by_episode[episode_id]:
            self._schedule_episode_finalize_after_grace(episode_id)

    def _compute_next_video_index(self) -> int:
        """
        Scan current videos dir and return the next integer index.
        Accepts files named like '1.webm', '2.mp4', etc.
        If directory is empty (typical after clear), returns 1.
        """
        if not self._demo_videos_dir:
            return 1
        max_idx = 0
        try:
            for p in self._demo_videos_dir.iterdir():
                if not p.is_file():
                    continue
                m = re.match(r"^(\d+)\.[A-Za-z0-9]+$", p.name)
                if m:
                    max_idx = max(max_idx, int(m.group(1)))
        except Exception:
            pass
        return (max_idx + 1) if max_idx > 0 else 1

    def next_video_filename(self, ext: str) -> tuple[str, int]:
        """Return ('{index}{ext}', index) and atomically increment the counter."""
        if not ext.startswith("."):
            ext = "." + ext
        with self._video_index_lock:
            idx = self._video_index
            self._video_index += 1
        return f"{idx}{ext}", idx

    def find_show_video_by_id(self, video_id: int | str) -> tuple[Path | None, str | None]:
        """
        VP9-only: resolve <id>.webm inside the show_videos_dir and return its path + mime.
        """
        vid = str(video_id).strip()
        if not vid.isdigit() or not self._show_videos_dir:
            return None, None

        p = self._show_videos_dir / f"{vid}.webm"
        if not p.is_file():
            return None, None

        mime = mimetypes.guess_type(str(p))[0] or "video/webm"
        return p, mime

    # --- NEW: helper to resolve the latest .webm by numeric filename ---
    def find_latest_show_video(self) -> tuple[Path | None, str | None]:
        """
        Return (path, id_str) of the latest .webm in _show_videos_dir.
        Files must be named like '<number>.webm' (e.g., 1.webm, 2.webm).
        """
        try:
            d = self._show_videos_dir
            if not d:
                return None, None
            latest_path = None
            latest_id = None
            for p in d.iterdir():
                if not (p.is_file() and p.suffix.lower() == ".webm"):
                    continue
                stem = p.stem.strip()
                if not stem.isdigit():
                    continue
                if latest_id is None or int(stem) > int(latest_id):
                    latest_path, latest_id = p, stem
            return latest_path, latest_id
        except Exception:
            return None, None

    def get_demo_video_config(self) -> dict:
        """
        Small, stable contract the frontend can consume.
        VP9-only: prefer .webm (VP9) and only accept VP9/WebM uploads.
        """
        cfg = {
            "enabled": bool(self.record_demo_videos),
            "task_name": (self.task_name() or "default"),
            "save_dir_abs": None,
            "save_dir_rel": None,
            "upload_url": "/api/upload-demo-video" if self.record_demo_videos else None,
            "preferred_extension": "webm",
            "preferred_mime": "video/webm",
            "suggest_canvas_capture": True,
            "filename_pattern": "{index}.{ext}",
            "sequence_start_index": 1,
            "reset_numbering_each_run": True,
            "accept_mimes": ["video/webm"]  # VP9-only
        }
        if self.record_demo_videos and self._demo_videos_dir:
            cfg["save_dir_abs"] = str(self._demo_videos_dir)
            cfg["save_dir_rel"] = self.rel_path_from_repo(self._demo_videos_dir)
        return cfg

    def set_active_episode(self, episode_id):
        """Mark which episode the outer robot loop is currently in (or None)."""
        with self.state_lock:
            self._active_episode_id = episode_id

    # --- Per-state VIEW snapshot cache (persist camera images to disk) ---
    def _persist_views_to_disk(self, episode_id: str, state_id: int, views_b64: dict[str, str]) -> dict[str, str]:
        """
        Persist base64 (data URL) JPEGs for each camera to disk.
        Returns a mapping: camera_name -> absolute file path.
        """
        if not views_b64:
            return {}
        out: dict[str, str] = {}
        try:
            d = self._episode_cache_dir(episode_id) / "views"
            d.mkdir(parents=True, exist_ok=True)
            for cam, data_url in views_b64.items():
                # Expect "data:image/jpeg;base64,....."
                if not isinstance(data_url, str):
                    continue
                idx = data_url.find("base64,")
                if idx == -1:
                    continue
                b64 = data_url[idx + len("base64,"):]
                try:
                    raw = base64.b64decode(b64)
                except Exception:
                    continue
                path = d / f"{state_id}_{cam}.jpg"
                with open(path, "wb") as f:
                    f.write(raw)
                out[cam] = str(path)
        except Exception as e:
            print(f"⚠️  failed to persist views ep={episode_id} state={state_id}: {e}")
        return out

    def _load_views_from_disk(self, view_paths: dict[str, str]) -> dict[str, str]:
        """
        Load per-camera JPEG files and return data URLs.
        """
        if not view_paths:
            return {}
        out: dict[str, str] = {}
        for cam, path in view_paths.items():
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                out[cam] = f"data:image/jpeg;base64,{b64}"
            except Exception:
                # Missing/removed file → skip this camera
                pass
        return out

    # --- Observation cache helpers (spill large 'observations' to disk) ---
    def _episode_cache_dir(self, episode_id: str) -> Path:
        d = self._obs_cache_root / str(episode_id)
        if not d.exists():
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        return d

    def _persist_obs_to_disk(self, episode_id: str, state_id: int, obs: dict) -> str | None:
        """
        Writes the observations dict to a single file for the state and returns the path.
        """
        try:
            p = self._episode_cache_dir(episode_id) / f"{state_id}.pt"
            # Tensors/ndarrays/py objects handled by torch.save
            torch.save(obs, p)
            return str(p)
        except Exception as e:
            print(f"⚠️  failed to persist obs ep={episode_id} state={state_id}: {e}")
            return None

    def load_obs_from_disk(self, path: str | None) -> dict:
        if not path:
            return {}
        try:
            return torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"⚠️  failed to load obs from {path}: {e}")
            return {}

    def _delete_obs_from_disk(self, path: str | None):
        if not path:
            return
        try:
            os.remove(path)
        except Exception:
            pass
    
    def _start_auto_label_worker(self):
        self.auto_label_worker_thread = Thread(target=self._auto_label_worker, daemon=True)
        self.auto_label_worker_thread.start()

    def _auto_label_worker(self):
        for critical_state_id in iter(self.auto_label_queue.get, None):
            self._auto_label(critical_state_id)
    
    def _auto_label(self, critical_state_id):
        '''
        Given critical_state_id, auto-labels noncritical states in the same episode before the critical state with:"
        1. The executed action of the previous important state
        2. If no previous important state exists, the joint positions of the first state in the episode
        '''
        with self.state_lock:
            episode_id = max(self.pending_states_by_episode.keys())

            episode_states = {
                **self.pending_states_by_episode[episode_id],
                **self.completed_states_by_episode[episode_id]
            }

            template_action = None

            previous_critical_id_in_episode = []
            for state_id in episode_states.keys():
                if episode_states[state_id]['critical'] \
                    and state_id < critical_state_id \
                    and len(episode_states[state_id]['actions']) > 0:
                    previous_critical_id_in_episode.append(state_id)

            if previous_critical_id_in_episode: # Previous critical states exist
                latest_critical_state = episode_states[max(previous_critical_id_in_episode)]
                template_action = latest_critical_state['actions'][0]
            else: # This is the first critical state in the episode
                first_state_id = min(episode_states.keys())
                first_state = episode_states[first_state_id]
                # Direct access to joint_positions and gripper in flattened structure
                joint_positions = first_state['joint_positions']
                gripper_action = first_state['gripper']
                goal_positions = []
                for joint_name in JOINT_NAMES:
                    joint_value = joint_positions[joint_name]
                    goal_positions.append(float(joint_value))

                goal_positions[-1] = 0.044 if gripper_action > 0 else 0.0
                template_action = torch.tensor(goal_positions, dtype=torch.float32)

            states_to_label = []
            for state_id, state_info in episode_states.items():
                if state_id < critical_state_id and not state_info['critical'] \
                and state_id not in self.completed_states_by_episode[episode_id]:
                    states_to_label.append(state_id)

            for state_id in states_to_label:
                state_info = episode_states[state_id]

                while state_info["responses_received"] < self.required_responses_per_state:
                    state_info["actions"].append(template_action.clone())
                    state_info['responses_received'] += 1

                all_actions = torch.cat(state_info["actions"][:self.required_responses_per_state], dim=0)
                        
                # Pad with inf values to match critical state shape
                missing_responses = self.required_responses_per_critical_state - self.required_responses_per_state
                action_dim = len(JOINT_NAMES)
                padding_size = missing_responses * action_dim
                padding = torch.full((padding_size,), float('nan'), dtype=torch.float32)
                all_actions = torch.cat([all_actions, padding], dim=0)

                state_info['action_to_save'] = all_actions

                # Save to completed_states buffer (for forming training set)
                if episode_id not in self.completed_states_buffer_by_episode:
                    self.completed_states_buffer_by_episode[episode_id] = {}
                self.completed_states_buffer_by_episode[episode_id][state_id] = state_info

                # Save to completed states (for monitoring)
                if episode_id not in self.completed_states_by_episode:
                    self.completed_states_by_episode[episode_id] = {}
                self.completed_states_by_episode[episode_id][state_id] = state_info

                del self.pending_states_by_episode[episode_id][state_id]

    # ---------- Episode → video ----------
    def load_main_cam_from_obs(self, obs: dict) -> np.ndarray | None:
        """
        Extract 'observation.images.cam_main' as RGB uint8 HxWx3; returns None if missing.
        """
        if not isinstance(obs, dict):
            return None
        for k in ("observation.images.cam_main", "observation.images.main", "observation.cam_main"):
            if k in obs:
                return self._to_uint8_rgb(obs[k])
        return None

    # --- NEW: prompts + logging + one-time context run ---------------------------------

    def _prompts_root_dir(self) -> Path:
        """Root folder containing prompts/."""
        return (Path(__file__).resolve().parent / ".." / "prompts").resolve()

    def task_name(self) -> str:
        """Prompt placeholder task name (from --task-name)."""
        return (self.prompt_task_name or "").strip()

    def _task_dir(self, task_name: str | None = None) -> Path:
        tn = task_name or self.task_name()
        return (self._prompts_root_dir() / tn).resolve()
    
    # --- NEW: helpers for critical-state image sequence ---
    def _compute_next_prompt_seq_index(self) -> int:
        """
        Scan the target directory and return next numeric index (1-based).
        Accepts files like 000001.jpg / 42.png / 7.jpeg, ignoring non-numeric stems.
        """
        nums = []
        for p in self._prompt_seq_dir.iterdir():
            if not p.is_file():
                continue
            m = re.match(r"^(\d+)$", p.stem)
            if m:
                nums.append(int(m.group(1)))
        return (max(nums) + 1) if nums else 1

    def _parse_description_bank_entries(self, file_path: str) -> list[dict]:
        """
        Read description bank from file. Each line is a text prompt.
        Line number corresponds to video number.
        Returns: [{"id": int, "text": "<line content>", "full": "<line content>"}]
        """
        entries = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line_content = line.strip()
                    if line_content:  # Skip empty lines
                        entries.append({
                            "id": line_num,
                            "text": line_content,
                            "full": line_content
                        })
        except FileNotFoundError:
            print(f"Description bank file not found: {file_path}")
        except Exception as e:
            print(f"Error reading description bank file {file_path}: {e}")
        
        return entries

    def get_description_bank(self) -> dict:
        """
        Return both the raw description-bank text and its parsed entries.
        Reads from prompts/{task-name}/descriptions.txt where each line is a text prompt.
        Line number corresponds to video number.
        """
        task_name = self.task_name() or ""
        if not task_name:
            print("Warning: No task name set, cannot load description bank")
            return {"raw_text": "", "entries": []}
        
        # Construct file path: prompts/{task-name}/descriptions.txt
        file_path = self._task_dir(task_name) / "descriptions.txt"
        
        # Read raw text for compatibility
        raw_text = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        except FileNotFoundError:
            print(f"Description bank file not found: {file_path}")
        except Exception as e:
            print(f"Error reading description bank file {file_path}: {e}")
        
        return {
            "raw_text": raw_text,
            "entries": self._parse_description_bank_entries(str(file_path))
        }

    def set_events(self, events):
        """Set the events object for keyboard-like control functionality"""
        self.events = events

    def _schedule_episode_finalize_after_grace(self, episode_id: int):
        delay = self.episode_finalize_grace_s
        timer = Timer(delay, self._finalize_episode_if_still_empty, args=(episode_id,))
        timer.daemon = True
        self._episode_finalize_timers[episode_id] = timer
        timer.start()

    def _finalize_episode_if_still_empty(self, episode_id: int):
        """
        Timer callback
        """
        with self.state_lock:
            self._episode_finalize_timers.pop(episode_id, None)

            if self.pending_states_by_episode.get(episode_id):
                # New states has become pending in the episode
                return
            
            self.episodes_completed.add(episode_id) # for monitoring

            buffer = self.completed_states_buffer_by_episode[episode_id]
            self.save_episode(buffer)

            del self.completed_states_buffer_by_episode[episode_id]
    
    ### ---Dataset Management---
    def save_episode(self, buffer):
        for state_id in sorted(buffer.keys()):
            state = buffer[state_id]
            obs = self.load_obs_from_disk(state['obs_path'])
            frame = {**obs, "action": state["action_to_save"], "task": state["task_text"]}
            self.dataset.add_frame(frame)
            self._delete_obs_from_disk(state.get("obs_path"))

        self.dataset.save_episode()

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

        # For UI fallback and dataset writes, always use cfg.single_task
        self.task_text = getattr(cfg, "single_task", None)
        
        # Update dataset action shape to accommodate crowd responses
        self._update_dataset_action_shape()
    
    def _update_dataset_action_shape(self):
        """Update the dataset's action feature shape to include crowd responses dimension"""
        if self.dataset is not None and "action" in self.dataset.features:
            from datasets import Sequence, Value, Features
            from lerobot.common.datasets.utils import get_hf_features_from_features
            
            original_action_dim = self.dataset.features["action"]["shape"][-1]  # Get the last dimension (joint count)
            new_action_shape = (self.required_responses_per_critical_state * original_action_dim,)
            
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

            print(f"📐 Updated dataset action shape to {new_action_shape} (crowd_responses={self.required_responses_per_critical_state}, joints={original_action_dim})")

    ### ---Camera Management---
    def init_cameras(self):
        """Open only *webcams* (skip RealSense nodes) once; skip any that fail."""
        self.cams = getattr(self, "cams", {})
        for name, idx in CAM_IDS.items():
            # Only attempt indices that look like webcams
            if not _is_webcam_idx(idx):
                print(f"⏭️  skipping '{name}' (/dev/video{idx}) - not a webcam")
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
                self._latest_jpeg[name] = self.encode_jpeg_base64(rgb)
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

    def snapshot_latest_views(self) -> dict[str, str]:
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
    
    def state_to_json(self, state: dict) -> dict:
        """
        Build the JSON payload for the labeling frontend:
        - If the state contains 'view_paths', load the state-aligned JPEGs from disk (correct behavior).
        - Otherwise, fall back to the latest live previews.
        Also attach static camera models/poses.
        """
        if not state:
            return {}
        out = dict(state)  # shallow copy (we'll remove internal fields)
        
        # Remove tensor fields that frontend doesn't need
        out.pop("actions", None)
        
        # Remove internal/disk paths that shouldn't be exposed to client
        obs_path = out.pop("obs_path", None)  # don't expose obs cache paths
        
        # Prefer state-aligned snapshots if available
        views = {}
        view_paths = out.pop("view_paths", None)  # don't expose file paths to the client
        if isinstance(view_paths, dict) and view_paths:
            views = self._load_views_from_disk(view_paths)
        # Fallback to live previews (older states or missing files)
        if not views:
            views = self.snapshot_latest_views()
        out["views"] = views
        out["camera_poses"] = self._camera_poses
        out["camera_models"] = self._camera_models
        out["gripper_tip_calib"] = self._gripper_tip_calib
        
        # --- NEW: attach example video URL (direct file URL; byte-range capable) ---
        if self.show_demo_videos:
            # Prefer a VLM-selected clip if available and present  
            video_id = state.get("video_prompt")
            chosen_url = None
            if video_id is not None:
                p, _ = self.find_show_video_by_id(video_id)
                if p:
                    chosen_url = f"/api/show-videos/{video_id}"  # serves the exact id

            # Fallback: latest available .webm
            if not chosen_url:
                lp, lid = self.find_latest_show_video()
                if lp and lid:
                    # Stable "latest" URL for the player; resolves dynamically on the server
                    chosen_url = "/api/show-videos/latest.webm"

            if chosen_url:
                out["example_video_url"] = chosen_url
        
        return out
    
    def encode_jpeg_base64(self, img_rgb: np.ndarray, quality: int | None = None) -> str:
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
    
    # --- Observation image streaming (background encoder) ---
    def _start_obs_stream_worker(self):
        if self._obs_img_running:
            return
        self._obs_img_running = True
        self._obs_img_thread = Thread(target=self._obs_stream_worker, daemon=True)
        self._obs_img_thread.start()

    def _to_uint8_rgb(self, arr) -> np.ndarray | None:
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
                    self._latest_obs_jpeg[name] = self.encode_jpeg_base64(rgb)
            except Exception:
                pass
            finally:
                try:
                    self._obs_img_queue.task_done()
                except Exception:
                    pass

    def _push_obs_view(self, name: str, img):
        """Enqueue an observation image for background JPEG encoding; drop if queue is full."""
        if img is None:
            return
        try:
            self._obs_img_queue.put_nowait((name, img))
        except queue.Full:
            # Drop frame to avoid backpressure on add_state
            pass
    
    def demote_earlier_unanswered_criticals(self, current_state_id, episode_id):
        '''
        Demote critical states before state_id in episode with episode_id to non-critical
        '''
        for state_id in self.pending_states_by_episode[episode_id].keys():
            if state_id < current_state_id \
                  and self.pending_states_by_episode[episode_id][state_id]['critical'] \
                  and not self.pending_states_by_episode[episode_id][state_id]['actions']:
                self.pending_states_by_episode[episode_id][state_id]['critical'] = False

    # --- State Management ---
    def add_state(self,
                  joint_positions: dict,
                  gripper_motion: int = None,
                  obs_dict: dict[str, torch.Tensor] = None,
                  episode_id: str = None,
                  left_carriage_external_force: float | None = None):
        '''
        Called by lerobot code to add states to backend.
        '''
        joint_positions_float = {k: float(v) for k, v in joint_positions.items()}

        state_id = self.next_state_id
        self.next_state_id += 1

        # Persist views to disk to avoid storing in memory
        view_paths = self._persist_views_to_disk(episode_id, state_id, self.snapshot_latest_views()) # legacy
        
        # Persist obs to disk
        obs_dict_deep_copy = {}
        for key, value in obs_dict.items():
            obs_dict_deep_copy[key] = value.clone().detach()
        obs_path = self._persist_obs_to_disk(episode_id, state_id, obs_dict_deep_copy)
        del obs_dict_deep_copy

        # Push obs to monitoring frontend
        self._push_obs_view("obs_main",  obs_dict.get("observation.images.cam_main"))
        self._push_obs_view("obs_wrist", obs_dict.get("observation.images.cam_wrist"))
        
        state_info = {
            # Identity
            "state_id": state_id,
            "episode_id": episode_id,

            # Robot state
            "joint_positions": joint_positions_float,
            "gripper": gripper_motion,
            "controls": ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper'], # legacy, will remove
            "left_carriage_external_force": left_carriage_external_force,

            # Observations
            "obs_path": obs_path,

            # Views
            "view_paths": view_paths,

            # Labels
            "actions": [],
            "responses_received": 0,

            # Critical state fields
            "critical": False,
            "prompt_ready": False,
            "text_prompt": None, # replaces flex_text_prompt
            "video_prompt": None, # replaces flex_video_id

            # Task
            "task_text": self.task_text

            # No other fields; segmentation, and all others, no longer supported\
        }

        with self.state_lock:
            # Initialize episode containers if needed
            if episode_id not in self.pending_states_by_episode:
                self.pending_states_by_episode[episode_id] = {}
                self.completed_states_by_episode[episode_id] = {}
                
            # Add state to pending states
            self.pending_states_by_episode[episode_id][state_id] = state_info

            self.current_serving_episode = episode_id

    def set_last_state_to_critical(self):
        with self.state_lock:
            
            if self.pending_states_by_episode:
                latest_episode_id = max(self.pending_states_by_episode.keys())
                episode_states = self.pending_states_by_episode[latest_episode_id]
                if episode_states:
                    latest_state_id = max(episode_states.keys())
                else:
                    return
            else:
                # self.pending_states_by_episode hasn't been populated yet
                return
            
            info = self.pending_states_by_episode[latest_episode_id][latest_state_id]
            if info['critical']:
                # Already set
                return
            info['critical'] = True

            self.demote_earlier_unanswered_criticals(latest_state_id, latest_episode_id)
            self.auto_label_previous_states(latest_state_id)

    def auto_label_previous_states(self, critical_state_id):
        self.auto_label_queue.put_nowait(critical_state_id)
    
    def get_latest_state(self) -> dict:
        """
        Get a pending state from current serving episode
        We only implement crowd mode, meaning that we serve the last state
        of the last episode always.
        """

        with self.state_lock:
            episode_id = self.current_serving_episode
            state_id = self.next_state_id - 1

            if episode_id not in self.pending_states_by_episode \
            or state_id not in self.pending_states_by_episode[episode_id]\
            or not self.pending_states_by_episode[episode_id][state_id]['critical']:
                # No pending critical states left
                return {
                    "status": "no_pending_states",
                    "blocked_critical_states": False
                }
            
            state_info = self.pending_states_by_episode[episode_id][state_id]

            if state_info['critical'] and not state_info['prompt_ready']:
                # There are pending states but no ready states

                return {
                    "status": "no_ready_states",
                    "blocked_critical_states": True,
                }
            
            # Return the latest state for labeling
            return state_info.copy()

    def record_response(self, response_data: dict):
        '''
        Record a response for a specific state. Handles all the side-effects.
        '''

        with self.state_lock:
            state_id = response_data['state_id']
            episode_id = response_data['episode_id']

            if episode_id not in self.pending_states_by_episode or \
            state_id not in self.pending_states_by_episode[episode_id]:
                # State already fully labeled
                return

            state_info = self.pending_states_by_episode[episode_id][state_id]

            required_responses = self.required_responses_per_critical_state if state_info['critical'] else self.required_responses_per_state
            
            joint_positions = response_data['joint_positions']
            gripper_action = response_data['gripper']

            state_info["responses_received"] += 1

            goal_positions = []
            for joint_name in JOINT_NAMES:
                joint_value = joint_positions[joint_name]
                goal_positions.append(float(joint_value[0]))

            goal_positions[-1] = 0.044 if gripper_action > 0 else 0.0
            goal_positions = torch.tensor(goal_positions, dtype=torch.float32)
            state_info['actions'].append(goal_positions)

            # Autofill
            if state_info["critical"] and self.autofill_critical_states:
                remaining = state_info["responses_received"]
                clones_to_add = min(self.num_autofill_actions - 1, remaining)
                for _ in range(clones_to_add):
                    state_info["actions"].append(goal_positions.clone())
                state_info['responses_received'] += clones_to_add
                
            # Handle completion
            if state_info['responses_received'] >= required_responses:
                if state_info['critical'] and state_id == self.next_state_id - 1:
                    # Choose action to execute (a_execute) at random
                    # Shift chosen action to the front of the array
                    a_execute_index = random.randint(0, required_responses - 1)
                    state_info["actions"][0], state_info["actions"][a_execute_index] = state_info["actions"][a_execute_index], state_info["actions"][0]
                    self.latest_goal = state_info["actions"][:required_responses][0]

                all_actions = torch.cat(state_info["actions"][:required_responses], dim=0)

                if required_responses < self.required_responses_per_critical_state:
                    # Pad unimportant states's action tensor
                    missing_responses = self.required_responses_per_critical_state - required_responses
                    action_dim = len(JOINT_NAMES)
                    padding_size = missing_responses * action_dim
                    padding = torch.full((padding_size,), float('nan'), dtype=torch.float32)
                    all_actions = torch.cat([all_actions, padding], dim=0)

                state_info['action_to_save'] = all_actions

                # Save to completed states buffer (for forming training set)
                if episode_id not in self.completed_states_buffer_by_episode:
                    self.completed_states_buffer_by_episode[episode_id] = {}
                self.completed_states_buffer_by_episode[episode_id][state_id] = state_info

                # Save to completed states (for monitoring)
                if episode_id not in self.completed_states_by_episode:
                    self.completed_states_by_episode[episode_id] = {}
                self.completed_states_by_episode[episode_id][state_id] = state_info

                # Remove from pending
                del self.pending_states_by_episode[episode_id][state_id]

                # Handle episode completion
                if not self.pending_states_by_episode[episode_id]:
                    self._schedule_episode_finalize_after_grace(episode_id)
    
    def get_pending_states_info(self) -> dict:
        """Get episode-based state information for monitoring"""
        with self.state_lock:
            episodes_info = {}
            total_pending = 0
            
            # Include episodes that have either pending states OR completed states (so completed states remain visible)
            all_episode_ids = set(self.pending_states_by_episode.keys()) | set(self.completed_states_by_episode.keys())
            
            # Process each episode
            for episode_id in sorted(all_episode_ids):
                episode_states = {}
                
                # Add pending states from this episode
                if episode_id in self.pending_states_by_episode:
                    for state_id, info in self.pending_states_by_episode[episode_id].items():
                        required_responses = (
                            self.required_responses_per_critical_state
                            if info.get('critical', False)
                            else self.required_responses_per_state
                        )
                        # --- NEW: compute prompt presence for pending states ---
                        _txt = info.get("text_prompt")  # Updated field name
                        has_flex_text = bool(str(_txt or "").strip())
                        _vid = info.get("video_prompt")  # Updated field name
                        has_flex_video = (_vid is not None)

                        episode_states[state_id] = {
                            "responses_received": info["responses_received"],
                            "responses_needed": required_responses - info["responses_received"],
                            "critical": bool(info.get("critical", False)),
                            "has_flex_text": has_flex_text,
                            "has_flex_video": has_flex_video,
                            # Legacy aliases to avoid breaking older monitor UI
                            "has_vlm_text": has_flex_text,
                            "has_video_id": has_flex_video,
                        }
                        total_pending += 1
                
                # Add completed states from this episode
                if episode_id in self.completed_states_by_episode:
                    for state_id, info in self.completed_states_by_episode[episode_id].items():
                        # --- NEW: compute prompt presence for completed states ---
                        _txt = info.get("text_prompt")  # Updated field name
                        has_flex_text = bool(str(_txt or "").strip())
                        _vid = info.get("video_prompt")  # Updated field name
                        has_flex_video = (_vid is not None)

                        episode_states[state_id] = {
                            "responses_received": info["responses_received"],
                            "responses_needed": 0,  # Completed
                            "critical": bool(info.get("critical", False)),
                            "has_flex_text": has_flex_text,
                            "has_flex_video": has_flex_video,
                            "has_vlm_text": has_flex_text,   # legacy
                            "has_video_id": has_flex_video,  # legacy
                        }
                
                episodes_info[episode_id] = {
                    "states": episode_states,
                    "pending_count": len(self.pending_states_by_episode.get(episode_id, {})),
                    "completed_count": len(self.completed_states_by_episode.get(episode_id, {})),
                    "is_current_serving": episode_id == self.current_serving_episode,
                    "is_completed": episode_id in self.episodes_completed,
                    # --- NEW: tell the monitor if this episode is waiting for an explicit save
                    "pending_save": episode_id in self._episodes_pending_save
                }
            
            return {
                "total_pending": total_pending,
                "current_serving_episode": self.current_serving_episode,
                "required_responses_per_state": self.required_responses_per_state,
                "required_responses_per_critical_state": self.required_responses_per_critical_state,
                "episodes": episodes_info
            }
    
    # --- Goal Management ---
    def get_latest_goal(self) -> dict | None:
        """Get and clear the latest goal (for robot loop to consume)"""
        goal = self.latest_goal
        self.latest_goal = None
        return goal
    
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
    
    def _calib_dir(self) -> Path:
        base_dir = Path(__file__).resolve().parent
        return (base_dir / ".." / "calib").resolve()

    def _load_gripper_tip_calibration(self) -> dict:
        """
        Load manual gripper tip calibration from ../calib/manual_gripper_tips.json
        Returns {"left":{"x":..,"y":..,"z":..}, "right":{...}}.
        """
        p = self._calib_dir() / "manual_gripper_tips.json"
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Minimal validation and float casting
        def _clean(side):
            s = data[side]
            return {
                "x": float(s["x"]),
                "y": float(s["y"]),
                "z": float(s["z"]),
            }
        return {"left": _clean("left"), "right": _clean("right")}

    def save_gripper_tip_calibration(self, calib: dict) -> str:
        """
        Save {"left":{"x","y","z"},"right":{"x","y","z"}} to ../calib/manual_gripper_tips.json
        Update in-memory self._gripper_tip_calib so the next /api/get-state reflects it.
        Returns the written path as a string.
        """
        # sanitize + cast
        def _want(side):
            s = (calib.get(side) or {})
            return {"x": float(s["x"]), "y": float(s["y"]), "z": float(s["z"])}
        try:
            cleaned = {"left": _want("left"), "right": _want("right")}
        except Exception as e:
            raise ValueError(f"invalid gripper_tip_calib payload: {e}")
        p = self._calib_dir()
        p.mkdir(parents=True, exist_ok=True)
        path = p / "manual_gripper_tips.json"
        with self._calib_lock:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(cleaned, f, indent=2)
                self._gripper_tip_calib = cleaned  # live update
            except Exception as e:
                raise IOError(f"failed to write {path}: {e}")
        return str(path)