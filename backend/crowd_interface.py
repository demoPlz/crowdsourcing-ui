from __future__ import annotations
import os
import cv2
import shutil
import tempfile
import numpy as np
import time
import torch
import base64
import random
import queue
import json
import traceback
import datasets
import re
# --- VLM (Azure OpenAI GPT-5) integration ------------------------------------
import mimetypes

def _safe_int(v, default):
    try: return int(v)
    except Exception: return default

from flask import Flask, jsonify, Response
from flask_cors import CORS
from flask import request, make_response
from pathlib import Path
from threading import Thread, Lock, Timer
from math import cos, sin
import math

# --- URDF FK & SAM2 segmentation ---
from PIL import Image

try:
    from urdfpy import URDF
except Exception:
    URDF = None

try:
    from transformers import Sam2Processor, Sam2Model
except Exception:
    Sam2Processor = None
    Sam2Model = None

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
                 use_vlm_prompt: bool = False,
                 use_manual_prompt: bool = False,
                 leader_mode: bool = False,
                 n_leaders: int | None = None,
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
                 # --- NEW ---
                 save_vlm_logs: bool = False,
                 vlm_logs_dir: str | None = None):

        # --- UI prompt mode (simple vs VLM vs MANUAL) ---
        self.use_vlm_prompt = bool(use_vlm_prompt or int(os.getenv("USE_VLM_PROMPT", "0")))
        self.use_manual_prompt = bool(use_manual_prompt or int(os.getenv("USE_MANUAL_PROMPT", "0")))
        if self.use_vlm_prompt and self.use_manual_prompt:
            raise ValueError("use_vlm_prompt and use_manual_prompt are mutually exclusive")

        self._vlm_enabled = False  # becomes True only if VLM is requested AND configured

        # --- Leader Mode controls ---
        self.leader_mode = bool(leader_mode)
        try:
            self.n_leaders = int(n_leaders) if (n_leaders is not None) else 1
        except Exception:
            self.n_leaders = 1
        if self.n_leaders < 1:
            self.n_leaders = 1
        # If enabled without VLM or manual prompt, disable with a warning (CLI should prevent this already)
        if self.leader_mode and not (self.use_vlm_prompt or self.use_manual_prompt):
            print("⚠️  --leader-mode ignored because neither --use-vlm-prompt nor --use-manual-prompt is enabled.")
            self.leader_mode = False
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
        
        # Clamp n_leaders to required_responses_per_critical_state
        if self.n_leaders > self.required_responses_per_critical_state:
            print(f"⚠️  n_leaders={self.n_leaders} > required_responses_per_critical_state={self.required_responses_per_critical_state}; clamping.")
            self.n_leaders = self.required_responses_per_critical_state
        
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
        self.episode_finalize_grace_s = float(os.getenv("EPISODE_FINALIZE_GRACE_S", "2"))
        self._episode_finalize_timers: dict[str, Timer] = {}


        # Precompute immutable views and camera poses to avoid per-tick allocations
        # Teardown fence
        self._shutting_down = False
        # Start the auto-labeling worker thread
        self._start_auto_label_worker()

        self._exec_gate_by_session: dict[str, dict] = {}

        self._active_episode_id = None
        self._start_obs_stream_worker()

        # --- URDF model / FK for backend gripper center (same link as frontend) ---
        self._urdf_model = None
        self._urdf_joint_names = set()
        # default path mirrors your frontend URDF; override with env URDF_PATH if different on backend
        self._urdf_path = os.getenv(
            "URDF_PATH",
            str((Path(__file__).resolve().parent / ".." / "public" / "assets" / "trossen_arm_description" /
                 "urdf" / "generated" / "wxai" / "wxai_base.urdf").resolve())
        )
        self._urdf_package_name = os.getenv("URDF_PACKAGE_NAME", "trossen_arm_description")
        # Default root: walk up from the URDF file to the '<package_name>' folder
        urdf_file = Path(self._urdf_path).resolve()
        default_pkg_root = None
        for anc in urdf_file.parents:
            if anc.name == self._urdf_package_name:
                default_pkg_root = str(anc)
                break
        if default_pkg_root is None:
            # Fallback guess: .../trossen_arm_description/urdf/generated/wxai/wxai_base.urdf → parents[3]
            try:
                default_pkg_root = str(urdf_file.parents[3])
            except Exception:
                default_pkg_root = str(urdf_file.parent)

        self._urdf_package_root = os.getenv(
            "URDF_PACKAGE_ROOT",
            default_pkg_root
        )
        self._urdf_ee_link_name = os.getenv("EE_LINK_NAME", "ee_gripper_link")
        self._load_urdf_model()

        # --- SAM2 segmentation (views-only) ---
        self._sam2_model = None
        self._sam2_processor = None
        self._sam2_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._sam2_model_id = os.getenv("SAM2_MODEL_ID", "facebook/sam2.1-hiera-tiny")

        # track masks on disk: episode_id -> { state_id -> {view_name -> mask_path_png} }
        self._segmentation_paths_by_episode: dict[str, dict[int, dict[str, str]]] = {}

        # segmentation tunables
        self._seg_dark_patch_radius = int(os.getenv("SEG_DARK_PATCH_RADIUS", "10"))     # px
        self._seg_dark_mean_thresh = float(os.getenv("SEG_DARK_MEAN_THRESH", "55.0"))  # 0..255

        # --- NEW: "flat gray" guard (RGB in [min,max] and channels nearly equal) ---
        self._seg_gray_patch_radius = int(os.getenv("SEG_GRAY_PATCH_RADIUS", str(self._seg_dark_patch_radius)))
        self._seg_gray_min          = float(os.getenv("SEG_GRAY_MIN", "161.0"))
        self._seg_gray_max          = float(os.getenv("SEG_GRAY_MAX", "200.0"))
        self._seg_gray_delta        = float(os.getenv("SEG_GRAY_DELTA", "10.0"))

        # --- Segmentation gating: only segment when gripper is "grasped" ---
        # Absolute force threshold in Newtons; if not available, we skip segmentation.
        self._grasp_force_thresh_N = float(os.getenv("SEG_GRASP_FORCE_THRESH_N", "50.0"))
        # Common telemetry keys to look for (first present numeric is used)
        self._grip_force_keys = (
            "left_carriage_external_force",
            "right_carriage_external_force",
            "gripper_force_n",
            "gripper_force",
        )

        # --- Multi-seed SAM2 prompting around gripper center ---
        self._seg_use_multi_seed   = bool(int(os.getenv("SEG_USE_MULTI_SEED", "0")))
        self._seg_seed_radius_px   = int(os.getenv("SEG_SEED_RADIUS_PX", "10"))   # base ring spacing, px
        self._seg_seed_rings       = int(os.getenv("SEG_SEED_RINGS", "1"))        # number of inner positive rings
        self._seg_seed_per_ring    = int(os.getenv("SEG_SEED_PER_RING", "8"))     # samples per ring
        self._seg_min_valid_seeds  = int(os.getenv("SEG_MIN_VALID_SEEDS", "0"))   # fallback to center if fewer

        # Optional negatives (one outer ring of 0-label clicks)
        self._seg_use_neg_ring     = bool(int(os.getenv("SEG_USE_NEG_RING", "0")))
        self._seg_neg_ring_scale   = float(os.getenv("SEG_NEG_RING_SCALE", "2.5"))  # outer ring radius multiplier vs base

        # Optional post-processing
        self._seg_mask_close_ksize = int(os.getenv("SEG_MASK_CLOSE_KSIZE", "0"))   # 0 to disable; else odd >=3

        # Multi-seed behavior for occlusion cases
        self._seg_center_negative  = bool(int(os.getenv("SEG_CENTER_NEGATIVE", "1")))  # center as NEG if dark/gray
        self._seg_ring_filter_mode = os.getenv("SEG_RING_FILTER_MODE", "loose")        # off | loose | strict
        self._seg_multimask        = bool(int(os.getenv("SEG_MULTIMASK", "1")))        # try multiple masks

        # Use a larger minimum ring radius so we click outside the gripper
        self._seg_seed_min_radius_px = int(os.getenv("SEG_SEED_MIN_RADIUS_PX", "12"))  # >= gripper radius in px

        # --- NEW: Important-state cam_main image sequence sink ---
        self.save_maincam_sequence = bool(save_maincam_sequence)
        self._prompt_seq_dir = Path(prompt_sequence_dir or "prompts/drawer/snapshots").resolve()
        self._prompt_seq_lock = Lock()
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
        self._video_ext_default = ".webm"
        if self.record_demo_videos:
            if demo_videos_dir:
                self._demo_videos_dir = Path(demo_videos_dir).resolve()
            else:
                # Default: prompts/demos/{task-name}/videos
                task_name = self.prompt_task_name or "default"
                repo_root = Path(__file__).resolve().parent / ".."
                self._demo_videos_dir = (repo_root / "prompts" / "demos" / task_name / "videos").resolve()
            
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

        # --- NEW: VLM logs ---
        self.save_vlm_logs = bool(save_vlm_logs or int(os.getenv("SAVE_VLM_LOGS", "0")))
        try:
            default_logs = (self._repo_root() / "output" / "vlm_logs").resolve()
        except Exception:
            default_logs = Path("output/vlm_logs").resolve()
        self._vlm_logs_dir = Path(vlm_logs_dir).resolve() if vlm_logs_dir else default_logs
        self._vlm_logs_written: set[tuple[str, int]] = set()

        if self.save_vlm_logs:
            try:
                self._vlm_logs_dir.mkdir(parents=True, exist_ok=True)
                print(f"📝 VLM logs → {self._vlm_logs_dir}")
            except Exception as e:
                print(f"⚠️ Could not prepare VLM logs directory '{self._vlm_logs_dir}': {e}")
                self.save_vlm_logs = False

        # --- Episode save behavior: datasets are always auto-saved after finalization ---
        # Manual save is only used for demo video recording workflow
        self._episodes_pending_save: set[str] = set()

    # ---------- Demo video config helpers (tell the frontend where to save) ----------
    def _repo_root(self) -> Path:
        """Root of the repo (backend assumes this file lives under <repo>/scripts or similar)."""
        return (Path(__file__).resolve().parent / "..").resolve()

    def _rel_path_from_repo(self, p: str | Path | None) -> str | None:
        if not p:
            return None
        try:
            rp = Path(p).resolve()
            return str(rp.relative_to(self._repo_root()))
        except Exception:
            # If not inside the repo root, return the basename as a safe hint.
            return os.path.basename(str(p))

    # ---------- Prompt-mode helpers (manual or VLM) ----------
    def _set_prompt_ready(self, state_info: dict, episode_id: int, state_id: int, text: str | None, video_id: int | None) -> None:
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

    def _next_video_filename(self, ext: str) -> tuple[str, int]:
        """Return ('{index}{ext}', index) and atomically increment the counter."""
        if not ext.startswith("."):
            ext = "." + ext
        with self._video_index_lock:
            idx = self._video_index
            self._video_index += 1
        return f"{idx}{ext}", idx

    def _find_show_video_by_id(self, video_id: int | str) -> tuple[Path | None, str | None]:
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
    def _find_latest_show_video(self) -> tuple[Path | None, str | None]:
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
            "task_name": (self._task_name() or "default"),
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
            cfg["save_dir_rel"] = self._rel_path_from_repo(self._demo_videos_dir)
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

    def _load_obs_from_disk(self, path: str | None) -> dict:
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

    def _purge_episode_cache(self, episode_id: str):
        """Remove the entire temp cache folder for an episode."""
        try:
            d = self._episode_cache_dir(episode_id)
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
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
    
    # ---------- Encoding helpers ----------
    def _file_to_data_url(self, path: str, mime: str) -> str:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def _image_file_to_data_url(self, path: str) -> str:
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        return self._file_to_data_url(path, mime)

    # ---------- Episode → video ----------
    def _load_main_cam_from_obs(self, obs: dict) -> np.ndarray | None:
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

    def _task_name(self) -> str:
        """Prompt placeholder task name (from --task-name)."""
        return (self.prompt_task_name or "").strip()

    def _task_dir(self, task_name: str | None = None) -> Path:
        tn = task_name or self._task_name()
        return (self._prompts_root_dir() / tn).resolve()

    def _demo_images_for_task(self, task_name: str | None = None) -> list[str]:
        """Return numerically sorted image file paths from prompts/demo/{task-name}/snapshots."""
        tn = task_name or self._task_name()
        demo_dir = (self._prompts_root_dir() / tn / "snapshots").resolve()
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        
        # Collect image files and sort them numerically by extracting numeric part
        image_files = []
        for p in demo_dir.iterdir():
            if p.suffix.lower() in exts and p.is_file():
                image_files.append(p)
        
        # Sort numerically by extracting the numeric part from filename
        def numeric_sort_key(path):
            # Extract numeric part from filename (e.g., "000001" from "000001.jpg")
            stem = path.stem
            # Find all digits in the filename
            import re
            numbers = re.findall(r'\d+', stem)
            if numbers:
                return int(numbers[0])  # Use first numeric sequence
            return float('inf')  # Put non-numeric files at the end
        
        sorted_files = sorted(image_files, key=numeric_sort_key)
        return [str(p) for p in sorted_files]

    def _load_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8").strip()

    def _count_sequence_descriptions(self, sequence_text: str) -> int:
        """
        Count the number of numbered descriptions in a sequence_description text.
        Expected format: "1. description", "2. description", etc.
        """
        count = 0
        lines = sequence_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            # Look for lines that start with a number followed by a period
            if re.match(r'^\d+\.', line):
                count += 1
        return count

    def _substitute_placeholders(self, template: str, task_name: str | None = None) -> str:
        """
        Replace every {x} in 'template' with contents of prompts/{task-name}/x.txt.
        Runs up to 3 passes to allow simple nested references.
        Special handling for {sequence_description} to add copying instructions.
        """
        tn = task_name or self._task_name()
        tdir = self._task_dir(tn)

        pat = re.compile(r"\{([A-Za-z0-9_\-]+)\}")
        out = template
        for _ in range(3):
            changed = False
            def repl(m):
                placeholder = m.group(1)
                fname = placeholder + ".txt"
                fpath = tdir / fname
                content = fpath.read_text(encoding="utf-8").strip()
                
                # Special handling for sequence_description to add copying instructions
                if placeholder == "sequence_description":
                    example_count = self._count_sequence_descriptions(content)
                    if example_count > 0:
                        copying_instruction = f"\nFor the first {example_count} frame pairs, DO NOT generate new descriptions. Instead, you MUST copy exactly these descriptions word-for-word from the examples above. Only after you have copied all {example_count} examples should you generate new descriptions for any remaining frames.\n\nExamples to copy:\n\n{content}"
                        return copying_instruction
                
                return content
            new_out = pat.sub(repl, out)
            if new_out != out:
                changed = True
                out = new_out
            if not changed:
                break
        return out

    def _substitute_dynamic_placeholders(self, template: str, episode_id: str, state_id: int) -> str:
        """
        Replace dynamic placeholders like [gripper_description] with context-specific text
        based on the current state information.
        """
        if not template or "[gripper_description]" not in template:
            return template
        
        # Get the state info to determine gripper status
        with self.state_lock:
            ep_states = self.pending_states_by_episode.get(episode_id, {})
            state_info = ep_states.get(state_id)
            if state_info is None:
                # Try completed states as fallback
                ep_completed = self.completed_states_buffer_by_episode.get(episode_id, {})
                state_info = ep_completed.get(state_id)
        
        if state_info is None:
            # No state info available, use default text
            gripper_desc = "The gripper status cannot be determined in this state."
        else:
            # Extract the state data
            state_data = state_info.get("state", {})
            
            # Check if gripper is grasped using existing method
            if self._gripper_is_grasped(state_data):
                gripper_desc = "The gripper has grasped onto the object in this state."
            else:
                gripper_desc = "The gripper is not grasped onto anything in this state."
        
        # Replace the placeholder
        result = template.replace("[gripper_description]", gripper_desc)
        return result

    def _load_prompt_with_subst(self, fname: str) -> str:
        """
        Load prompts/{fname} and apply {x} substitution using prompts/{task-name}/x.txt.
        """
        p = (self._prompts_root_dir() / fname).resolve()
        raw = self._load_text(p)
        return self._substitute_placeholders(raw, self._task_name())

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

    # --- PATCH: VLM modal helpers -----------------------------------------------

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
        task_name = self._task_name() or ""
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
    # --- END PATCH ---------------------------------------------------------------

    def set_events(self, events):
        """Set the events object for keyboard-like control functionality"""
        self.events = events

    def _schedule_episode_finalize_after_grace(self, episode_id: int):
        delay = self.episode_finalize_grace_s
        timer = Timer(delay, self._finalize_episode_if_still_empty, args=(episode_id,))
        timer.daemon = True
        self._episode_finalize_timers[episode_id] = timer
        timer.start()

    def _cancel_episode_finalize_timer(self, episode_id: int):
        t = self._episode_finalize_timers.pop(episode_id, None)
        if t:
            t.cancel()

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
            obs = self._load_obs_from_disk(state['obs_path'])
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

        # NEW: run the one-time context query (if enabled/configured)
        try:
            if self._vlm_enabled and self.use_vlm_prompt:
                self._run_vlm_context_once()
        except Exception as e:
            print(f"⚠️ Context VLM call failed: {e}")
        
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
    
    def _state_to_json(self, state: dict) -> dict:
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
            views = self._snapshot_latest_views()
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
                p, _ = self._find_show_video_by_id(video_id)
                if p:
                    chosen_url = f"/api/show-videos/{video_id}"  # serves the exact id

            # Fallback: latest available .webm
            if not chosen_url:
                lp, lid = self._find_latest_show_video()
                if lp and lid:
                    # Stable "latest" URL for the player; resolves dynamically on the server
                    chosen_url = "/api/show-videos/latest.webm"

            if chosen_url:
                out["example_video_url"] = chosen_url
        
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
    
    # --- Observation image streaming (background encoder) ---
    def _start_obs_stream_worker(self):
        if self._obs_img_running:
            return
        self._obs_img_running = True
        self._obs_img_thread = Thread(target=self._obs_stream_worker, daemon=True)
        self._obs_img_thread.start()

    def _stop_obs_stream_worker(self):
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

    def _push_obs_view(self, name: str, img):
        """Enqueue an observation image for background JPEG encoding; drop if queue is full."""
        if img is None:
            return
        try:
            self._obs_img_queue.put_nowait((name, img))
        except queue.Full:
            # Drop frame to avoid backpressure on add_state
            pass
    
    def _global_max_existing_state_id(self) -> int | None:
        """
        Return the newest state_id that EXISTS anywhere in memory:
        - pending states,
        - completed states (metadata),
        - completed states buffered for chronological add (manifest).
        """
        max_id = None
        # Pending
        for ep_states in self.pending_states_by_episode.values():
            if ep_states:
                k = max(ep_states.keys())
                max_id = k if max_id is None or k > max_id else max_id
        # Completed metadata
        for ep_done in self.completed_states_by_episode.values():
            if ep_done:
                k = max(ep_done.keys())
                max_id = k if max_id is None or k > max_id else max_id
        # Completed buffer (manifest entries awaiting add/save)
        for ep_buf in self.completed_states_buffer_by_episode.values():
            if ep_buf:
                k = max(ep_buf.keys())
                max_id = k if max_id is None or k > max_id else max_id
        return max_id
    
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
        view_paths = self._persist_views_to_disk(episode_id, state_id, self._snapshot_latest_views()) # legacy
        
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

    def _extract_grip_force_N(self, frontend_state: dict) -> float | None:
        """Return grip force in Newtons if available and numeric; otherwise None."""
        if not isinstance(frontend_state, dict):
            return None
        for k in self._grip_force_keys:
            v = frontend_state.get(k, None)
            if v is None:
                continue
            try:
                fv = float(v)
                if math.isfinite(fv):
                    return fv
            except Exception:
                continue
        return None

    def _gripper_is_grasped(self, frontend_state: dict) -> bool:
        """
        Define 'grasped' as force < -threshold (negative force indicates compression/grasping). 
        If force is missing/invalid, treat as NOT grasped.
        """
        f = self._extract_grip_force_N(frontend_state)
        return (f is not None) and (f < -self._grasp_force_thresh_N)

    # =================== URDF FK & SAM2 helpers ===================

    def _load_urdf_model(self):
        if URDF is None:
            print("⚠️  urdfpy not installed; set `pip install urdfpy` to enable backend FK.")
            return
        try:
            if not os.path.exists(self._urdf_path):
                print(f"⚠️  URDF file not found at: {self._urdf_path}")
                return

            # 1) Try with package_map so 'package://<pkg>/' resolves to the real folder
            package_map = {self._urdf_package_name: self._urdf_package_root}
            print(f"🔎 Loading URDF with package_map: {package_map}")
            try:
                self._urdf_model = URDF.load(self._urdf_path, package_map=package_map)
            except TypeError:
                # Older urdfpy might not accept package_map kw; try from_xml_file with it:
                self._urdf_model = URDF.from_xml_file(self._urdf_path, package_map=package_map)

        except Exception as e1:
            # 2) Fallback: rewrite package:// URIs to absolute paths in a temp file
            try:
                prefix = f"package://{self._urdf_package_name}/"
                with open(self._urdf_path, "r", encoding="utf-8") as f:
                    xml = f.read()
                if prefix in xml:
                    resolved_root = self._urdf_package_root.rstrip("/") + "/"
                    xml_resolved = xml.replace(prefix, resolved_root)
                    tmp_path = Path(tempfile.gettempdir()) / ("resolved_" + Path(self._urdf_path).name)
                    with open(tmp_path, "w", encoding="utf-8") as g:
                        g.write(xml_resolved)
                    print(f"🧩 Rewrote package URIs → {tmp_path}")
                    self._urdf_model = URDF.load(str(tmp_path))
                else:
                    raise  # No package:// in file; re-raise original
            except Exception as e2:
                print(f"⚠️  failed to load URDF even after rewrite: {e2}")
                print(f"    original error: {e1}")
                self._urdf_model = None
                return

        # Success path: record joint names, sanity logs
        try:
            self._urdf_joint_names = {j.name for j in self._urdf_model.joints}
        except Exception:
            self._urdf_joint_names = set()
        print(
            f"✓ URDF loaded from {self._urdf_path}\n"
            f"  package '{self._urdf_package_name}' → {self._urdf_package_root}\n"
            f"  joints detected: {len(self._urdf_joint_names)}"
        )


    def _gripper_center_from_joints(self, joint_positions: dict) -> np.ndarray | None:
        """
        Compute ee world position using URDF FK at the time the state was captured.
        Uses the link name 'ee_gripper_link' by default (same as frontend).
        World frame is the robot base frame (assumed aligned with Three.js world).
        """
        if self._urdf_model is None:
            return None
        try:
            # Build FK config with available joint entries; others default to 0
            cfg = {}
            for name, v in (joint_positions or {}).items():
                # incoming values might be scalars or [scalar]
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    vv = float(v[0])
                else:
                    vv = float(v)
                if name in self._urdf_joint_names:
                    cfg[name] = vv

            fk_all = self._urdf_model.link_fk(cfg=cfg)  # dict: Link -> 4x4
            target_link = None
            for L in self._urdf_model.links:
                if L.name == self._urdf_ee_link_name:
                    target_link = L
                    break
            if target_link is None:
                print(f"⚠️  URDF link '{self._urdf_ee_link_name}' not found")
                return None
            T = fk_all.get(target_link)
            if T is None:
                return None
            pos = np.asarray(T, dtype=np.float64)[:3, 3]
            return pos
        except Exception as e:
            print(f"⚠️  FK error: {e}")
            return None

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
         allow_headers=["Content-Type", "ngrok-skip-browser-warning", "X-Session-ID"],
         methods=["GET", "POST", "OPTIONS"])
    
    @app.route("/api/get-state")
    def get_state():
        if crowd_interface._shutting_down:
            return jsonify({}), 200
        
        state = crowd_interface.get_latest_state()
        
        # Check if this is a status response (no real state)
        if isinstance(state, dict) and state.get("status"):
            # Return status response directly without processing through _state_to_json
            response = jsonify(state)
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response
        
        # Process as a real state
        payload = crowd_interface._state_to_json(state)
        
        # Prefer text_prompt (manual or VLM), otherwise simple fallback
        text = payload.get("text_prompt")
        if isinstance(text, str) and text.strip():
            payload["prompt"] = text.strip()
        else:
            payload["prompt"] = f"Task: {crowd_interface.task_text or 'crowdsourced_task'}. What should the arm do next?"

        # NEW: Always tell the frontend what to do with demo videos
        payload["demo_video"] = crowd_interface.get_demo_video_config()
        
        # NEW: Add leader mode information for frontend delay logic
        payload["leader_mode"] = crowd_interface.leader_mode

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

    @app.route("/api/demo-video-config", methods=["GET"])
    def demo_video_config():
        """
        Lightweight config endpoint so the new frontend can fetch once on load.
        Mirrors the 'demo_video' object we also embed in /api/get-state.
        """
        try:
            return jsonify(crowd_interface.get_demo_video_config())
        except Exception as e:
            return jsonify({"enabled": False, "error": str(e)}), 500
    
    @app.route("/api/submit-goal", methods=["POST"])
    def submit_goal():
        try:
            if crowd_interface._shutting_down:
                return jsonify({"status": "shutting_down"}), 503
            
            # Validate request data
            data = request.get_json(force=True, silent=True)
            if data is None:
                return jsonify({"status": "error", "message": "Invalid JSON data"}), 400
            
            # Check for required fields
            required_fields = ['state_id', 'episode_id', 'joint_positions', 'gripper']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return jsonify({"status": "error", "message": f"Missing required fields: {missing_fields}"}), 400
            
            # Generate or retrieve session ID from request headers or IP
            session_id = request.headers.get('X-Session-ID', request.remote_addr or 'unknown')
            
            # Record this as a response to the correct state for this session
            # The frontend now includes state_id in the request data
            crowd_interface.record_response(data)
            return jsonify({"status": "ok"})
            
        except KeyError as e:
            print(f"❌ KeyError in submit_goal (missing data field): {e}")
            return jsonify({"status": "error", "message": f"Missing required field: {e}"}), 400
        except Exception as e:
            print(f"❌ Error in submit_goal endpoint: {e}")
            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route("/api/pending-states-info")
    def pending_states_info():
        """Debug endpoint to see pending states information"""
        info = crowd_interface.get_pending_states_info()
        return jsonify(info)
    
    # --- PATCH: Endpoints for monitor modal -------------------------------------

    @app.route("/api/description-bank", methods=["GET"])
    def api_description_bank():
        """
        Return the description bank for the current task as both:
          - 'entries': [{id, text, full}]
          - 'raw_text': the unparsed text (for debugging or custom parsing)
        """
        try:
            if crowd_interface._shutting_down:
                return jsonify({"ok": False, "error": "shutting_down"}), 503
            bank = crowd_interface.get_description_bank()
            return jsonify({"ok": True, "task_name": (crowd_interface._task_name() or "default"),
                            "entries": bank["entries"], "raw_text": bank["raw_text"]})
        except Exception as e:
            print(f"❌ /api/description-bank error: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500


    @app.route("/api/state-details", methods=["GET"])
    def api_state_details():
        """
        Query params: episode_id=<str|int>, state_id=<int>
        Returns:
          - maincam_data_url
          - is_critical
          - flex_text_prompt
          - flex_video_id
          - description_bank / description_bank_text
        """
        try:
            if crowd_interface._shutting_down:
                return jsonify({"ok": False, "error": "shutting_down"}), 503

            ep = request.args.get("episode_id", type=int)
            sid = request.args.get("state_id", type=int)
            if ep is None or sid is None:
                return jsonify({"ok": False, "error": "episode_id and state_id are required"}), 400

            # Defaults
            flex_text = ""
            flex_video_id = None
            is_imp = False
            obs_path = None

            with crowd_interface.state_lock:
                # Prefer pending
                p_ep = crowd_interface.pending_states_by_episode.get(ep, {})
                p_info = p_ep.get(sid)
                if p_info is not None:
                    is_imp = bool(p_info.get("critical", False))
                    obs_path = p_info.get("obs_path")
                    # Text: use new field name
                    flex_text = p_info.get("text_prompt") or ""
                    # Video id: use new field name  
                    raw_vid = p_info.get("video_prompt")
                    try:
                        flex_video_id = int(raw_vid) if raw_vid is not None else None
                    except Exception:
                        flex_video_id = None
                else:
                    # Completed metadata
                    c_ep = crowd_interface.completed_states_by_episode.get(ep, {})
                    c_meta = c_ep.get(sid)
                    if c_meta is None:
                        return jsonify({"ok": False, "error": f"state {sid} not found in episode {ep}"}), 404
                    is_imp = bool(c_meta.get("critical", False))  # Use consistent field name
                    flex_text = c_meta.get("text_prompt") or ""
                    raw_vid = c_meta.get("video_prompt")
                    try:
                        flex_video_id = int(raw_vid) if raw_vid is not None else None
                    except Exception:
                        flex_video_id = None
                    man = crowd_interface.completed_states_buffer_by_episode.get(ep, {}).get(sid)
                    if isinstance(man, dict):
                        obs_path = man.get("obs_path")

            # Load maincam image (if possible)
            maincam_url = None
            if obs_path:
                obs = crowd_interface._load_obs_from_disk(obs_path)
                img = crowd_interface._load_main_cam_from_obs(obs)
                if img is not None:
                    maincam_url = crowd_interface._encode_jpeg_base64(img)

            # Description bank
            bank = crowd_interface.get_description_bank()

            return jsonify({
                "ok": True,
                "episode_id": ep,
                "state_id": sid,
                "critical": is_imp,
                "text_prompt": flex_text,
                "video_prompt": flex_video_id,
                "maincam_data_url": maincam_url,
                "description_bank": bank["entries"],
                "description_bank_text": bank["raw_text"]
            })
        except Exception as e:
            print(f"❌ /api/state-details error: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/update-flex-selection", methods=["POST"])
    def api_update_flex_selection():
        """
        Body JSON:
        {
          "episode_id": <str or int>,
          "state_id": <int>,
          "video_prompt": <int>
          "text_prompt": <str>
        }
        """
        try:
            if crowd_interface._shutting_down:
                return jsonify({"ok": False, "error": "shutting_down"}), 503

            data = request.get_json(force=True, silent=True) or {}
            ep_raw = data.get("episode_id")
            if ep_raw is None:
                return jsonify({"ok": False, "error": "episode_id is required"}), 400
            ep = int(ep_raw)

            sid = data.get("state_id")
            if sid is None:
                return jsonify({"ok": False, "error": "state_id is required"}), 400
            sid = int(sid)

            vid = data.get("video_prompt")
            if vid is None:
                return jsonify({"ok": False, "error": "video_prompt is required"}), 400
            vid = int(vid)

            txt = (data.get("text_prompt") or "").strip()

            updated = False
            with crowd_interface.state_lock:
                # pending?
                p_ep = crowd_interface.pending_states_by_episode.get(ep, {})
                p_info = p_ep.get(sid)
                if p_info is not None:
                    crowd_interface._set_prompt_ready(p_info, ep, sid, txt if txt else None, vid)
                    updated = True
                else:
                    # completed metadata path  
                    c_ep = crowd_interface.completed_states_by_episode.get(ep, {})
                    c_info = c_ep.get(sid)
                    if c_info is not None:
                        # metadata mirrors - use new field names
                        if txt:
                            c_info["text_prompt"] = txt
                        c_info["video_prompt"] = vid
                        c_info["prompt_ready"] = True
                        updated = True

            if not updated:
                return jsonify({"ok": False, "error": f"state {sid} not found in episode {ep}"}), 404

            return jsonify({"ok": True, "episode_id": ep, "state_id": sid,
                            "video_prompt": vid, "text_prompt": txt or None})
        except Exception as e:
            print(f"❌ /api/update-flex-selection error: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/update-vlm-selection", methods=["POST"])  # legacy alias
    def api_state_vlm_selection():
        # Delegate to the new handler (accepts both field name variants)
        return api_update_flex_selection()
    
    # --- END PATCH ---------------------------------------------------------------
    
    @app.route("/api/monitor/latest-state", methods=["GET"])
    def monitor_latest_state():
        """
        Read-only monitoring endpoint for episode-based state monitoring.
        Avoid building a combined dict of all pending states on every call.
        """
        try:
            with crowd_interface.state_lock:
                current_episode = crowd_interface.current_serving_episode

                total_pending = 0
                newest_state_id = None
                newest_state_data = None
                newest_episode_id = None

                for ep_id, ep_states in crowd_interface.pending_states_by_episode.items():
                    n = len(ep_states)
                    total_pending += n
                    if n == 0:
                        continue
                    # Max by key without materializing a merged dict
                    ep_max_id = max(ep_states.keys())
                    if newest_state_id is None or ep_max_id > newest_state_id:
                        newest_state_id = ep_max_id
                        newest_state_data = ep_states[ep_max_id]
                        newest_episode_id = ep_id

                if total_pending == 0 or newest_state_data is None:
                    return jsonify({
                        "status": "no_pending_states",
                        "message": "No pending states.",
                        "views": crowd_interface._snapshot_latest_views(),  # still show previews
                        "total_pending_states": 0,
                        "current_serving_episode": current_episode,
                        "robot_moving": crowd_interface.is_robot_moving(),
                        "is_async_collection": crowd_interface.is_async_collection_mode(),
                        "is_resetting": crowd_interface.is_in_reset(),
                        "reset_countdown": crowd_interface.get_reset_countdown(),
                        "timestamp": time.time()
                    })

            # Build response outside the lock
            # newest_state_data IS the state info directly (flattened structure)
            monitoring_data = {
                "status": "success",
                "state_id": newest_state_id,
                "episode_id": newest_episode_id,
                "current_serving_episode": current_episode,
                "timestamp": time.time(),  # Use current time since we no longer store state timestamps
                "responses_received": newest_state_data["responses_received"],
                "responses_required": (
                    crowd_interface.required_responses_per_critical_state
                    if newest_state_data.get("critical", False)
                    else crowd_interface.required_responses_per_state
                ),
                "critical": newest_state_data.get("critical", False),
                "views": crowd_interface._snapshot_latest_views(),  # lightweight snapshot (pre-encoded)
                "joint_positions": newest_state_data.get("joint_positions", {}),
                "gripper": newest_state_data.get("gripper", 0),
                "robot_moving": crowd_interface.is_robot_moving(),
                "is_async_collection": crowd_interface.is_async_collection_mode(),
                "is_resetting": crowd_interface.is_in_reset(),
                "reset_countdown": crowd_interface.get_reset_countdown(),
                "total_pending_states": total_pending,
                "current_time": time.time()
            }

            response = jsonify(monitoring_data)
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response

        except Exception as e:
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
         # Allow multiplexing to gripper tip handler
        data = request.get_json(force=True, silent=True) or {}
        typ = (data.get("type") or "").strip().lower()
        if typ == "gripper_tips":
            calib = data.get("gripper_tip_calib") or {}
            # minimal validation
            if not isinstance(calib, dict) or "left" not in calib or "right" not in calib:
                return jsonify({"error": "gripper_tip_calib must include 'left' and 'right' {x,y,z}"}), 400
            try:
                out_path = crowd_interface.save_gripper_tip_calibration(calib)
                return jsonify({"status": "ok", "path": out_path})
            except (ValueError, IOError) as e:
                return jsonify({"error": str(e)}), 400
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
        
    @app.route("/api/save-gripper-tips", methods=["POST"])
    def save_gripper_tips():
        if crowd_interface._shutting_down:
            return jsonify({"status": "shutting_down"}), 503
        try:
            data = request.get_json(force=True, silent=True) or {}
            calib = data.get("gripper_tip_calib") or {}
            if not isinstance(calib, dict) or "left" not in calib or "right" not in calib:
                return jsonify({"error": "gripper_tip_calib must include 'left' and 'right' {x,y,z}"}), 400
            out_path = crowd_interface.save_gripper_tip_calibration(calib)
            return jsonify({"status": "ok", "path": out_path})
        except (ValueError, IOError) as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"unexpected error: {e}"}), 500
    
    @app.route("/api/upload-demo-video", methods=["POST"])
    def upload_demo_video():
        if crowd_interface._shutting_down:
            return jsonify({"status": "shutting_down"}), 503
        
        if not crowd_interface.record_demo_videos:
            return jsonify({"error": "Demo video recording is not enabled"}), 400
        
        try:
            if 'video' not in request.files:
                return jsonify({"error": "No video file provided"}), 400

            file = request.files['video']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400

            # Enforce WebM uploads (VP9-only system)
            if getattr(file, "mimetype", None) and "webm" not in file.mimetype.lower():
                return jsonify({"error": "Only WebM/VP9 uploads are accepted"}), 400

            metadata = {}
            if 'metadata' in request.form:
                try:
                    metadata = json.loads(request.form['metadata'])
                except:
                    pass

            # Always write sequential *.webm
            ext = ".webm"
            filename, index = crowd_interface._next_video_filename(ext)
            file_path = crowd_interface._demo_videos_dir / filename
            file.save(str(file_path))

            if metadata:
                metadata_path = (crowd_interface._demo_videos_dir / f"{index}.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)

            public_url = crowd_interface._maybe_upload_to_blob(str(file_path))

            return jsonify({
                "status": "success",
                "filename": filename,
                "path": str(file_path),
                "save_dir_rel": crowd_interface._rel_path_from_repo(file_path.parent),
                "public_url": public_url,
                "config": crowd_interface.get_demo_video_config(),
                "index": index
            })
            
        except Exception as e:
            print(f"❌ Error uploading demo video: {e}")
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/demo-videos/<filename>")
    def serve_demo_video(filename):
        """Serve demo video files for the frontend example video feature."""
        if crowd_interface._shutting_down:
            return jsonify({"status": "shutting_down"}), 503
        
        if not crowd_interface._demo_videos_dir:
            return jsonify({"error": "Demo videos directory not configured"}), 404
        
        try:
            # Sanitize filename to prevent directory traversal
            filename = os.path.basename(filename)
            file_path = crowd_interface._demo_videos_dir / filename
            
            if not file_path.exists():
                return jsonify({"error": "Video file not found"}), 404
            
            # Determine MIME type
            mime_type = mimetypes.guess_type(str(file_path))[0] or 'video/webm'
            
            # Create response with proper headers for video streaming
            response = make_response()
            response.headers['Content-Type'] = mime_type
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Cache-Control'] = 'public, max-age=3600'  # Cache for 1 hour
            
            # Read and return the file
            with open(file_path, 'rb') as f:
                response.data = f.read()
            
            return response
            
        except Exception as e:
            print(f"❌ Error serving demo video {filename}: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/show-videos/<video_id>")
    def serve_show_video(video_id):
        """
        Serve read-only example videos by numeric id from prompts/{task-name}/videos (or custom dir).
        This endpoint is independent of the recording feature and supports HTTP Range.
        """
        if crowd_interface._shutting_down:
            return jsonify({"status": "shutting_down"}), 503

        if not crowd_interface.show_demo_videos or not crowd_interface._show_videos_dir:
            return jsonify({"error": "Show demo videos is not enabled"}), 404

        # Sanitize; we only accept digits for ids.
        vid = "".join(c for c in str(video_id) if c.isdigit())
        if not vid:
            return jsonify({"error": "Invalid video id"}), 400

        file_path, mime = crowd_interface._find_show_video_by_id(vid)
        if not file_path:
            return jsonify({"error": "Video file not found"}), 404

        try:
            file_size = os.path.getsize(file_path)
            range_header = request.headers.get("Range", None)

            if range_header:
                # Format: "bytes=start-end"
                m = re.match(r"bytes=(\d+)-(\d*)", range_header)
                if m:
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else file_size - 1
                    end = min(end, file_size - 1)
                    if start > end or start >= file_size:
                        # RFC 7233
                        resp = Response(status=416)
                        resp.headers["Content-Range"] = f"bytes */{file_size}"
                        return resp

                    length = end - start + 1
                    with open(file_path, "rb") as f:
                        f.seek(start)
                        data = f.read(length)

                    rv = Response(data, 206, mimetype=mime, direct_passthrough=True)
                    rv.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
                    rv.headers["Accept-Ranges"] = "bytes"
                    rv.headers["Content-Length"] = str(length)
                    rv.headers["Cache-Control"] = "public, max-age=3600"
                    return rv

            # No Range: return full file
            with open(file_path, "rb") as f:
                data = f.read()
            rv = make_response(data)
            rv.headers["Content-Type"] = mime
            rv.headers["Content-Length"] = str(file_size)
            rv.headers["Accept-Ranges"] = "bytes"
            rv.headers["Cache-Control"] = "public, max-age=3600"
            return rv

        except Exception as e:
            print(f"❌ Error serving show video {video_id}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/show-videos/latest.webm")
    def serve_latest_show_video():
        """
        Serve the most recent .webm in the show_videos_dir with full HTTP Range support.
        Content-Type: video/webm
        Accept-Ranges: bytes
        """
        if crowd_interface._shutting_down:
            return jsonify({"status": "shutting_down"}), 503

        if not crowd_interface.show_demo_videos or not crowd_interface._show_videos_dir:
            return jsonify({"error": "Show demo videos is not enabled"}), 404

        # Resolve the latest numeric .webm (e.g., 1.webm, 2.webm, ...)
        latest_path, latest_id = crowd_interface._find_latest_show_video()
        if not latest_path or not latest_path.exists():
            return jsonify({"error": "No video file found"}), 404

        try:
            file_path = latest_path
            mime = "video/webm"  # force WebM for the player

            file_size = os.path.getsize(file_path)
            range_header = request.headers.get("Range", None)

            if range_header:
                # Format: "bytes=start-end"
                m = re.match(r"bytes=(\d+)-(\d*)", range_header)
                if m:
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else file_size - 1
                    end = min(end, file_size - 1)
                    if start > end or start >= file_size:
                        resp = Response(status=416)
                        resp.headers["Content-Range"] = f"bytes */{file_size}"
                        return resp

                    length = end - start + 1
                    with open(file_path, "rb") as f:
                        f.seek(start)
                        data = f.read(length)

                    rv = Response(data, 206, mimetype=mime, direct_passthrough=True)
                    rv.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
                    rv.headers["Accept-Ranges"] = "bytes"
                    rv.headers["Content-Length"] = str(length)
                    rv.headers["Cache-Control"] = "public, max-age=3600"
                    return rv

            # No Range header → return the whole file
            with open(file_path, "rb") as f:
                data = f.read()
            rv = make_response(data)
            rv.headers["Content-Type"] = mime
            rv.headers["Content-Length"] = str(file_size)
            rv.headers["Accept-Ranges"] = "bytes"
            rv.headers["Cache-Control"] = "public, max-age=3600"
            return rv

        except Exception as e:
            print(f"❌ Error serving latest show video: {e}")
            return jsonify({"error": str(e)}), 500

    # Streaming recording endpoints for canvas-based recording
    # In-memory storage for active recording sessions
    recording_sessions = {}  # recording_id -> {task_name, ext, chunks: [bytes]}
    
    @app.route("/api/record/start", methods=["POST"])
    def record_start():
        if crowd_interface._shutting_down:
            return jsonify({"status": "shutting_down"}), 503
        
        if not crowd_interface.record_demo_videos:
            return jsonify({"error": "Demo video recording is not enabled"}), 400
        
        try:
            data = request.get_json() or {}
            recording_id = data.get('recording_id')
            task_name = data.get('task_name') or crowd_interface._task_name() or 'default'
            ext = 'webm'   # VP9-only
            
            if not recording_id:
                return jsonify({"error": "missing recording_id"}), 400
            
            # Initialize session
            recording_sessions[recording_id] = {
                'task_name': task_name,
                'ext': ext,
                'chunks': [],
                'started_at': data.get('started_at'),
                'metadata': data
            }
            
            return jsonify({"ok": True})
            
        except Exception as e:
            print(f"❌ Error starting recording: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/record/chunk", methods=["POST"])
    def record_chunk():
        if crowd_interface._shutting_down:
            return jsonify({"status": "shutting_down"}), 503
        
        if not crowd_interface.record_demo_videos:
            return jsonify({"error": "Demo video recording is not enabled"}), 400
        
        try:
            recording_id = request.args.get('rid')
            seq = request.args.get('seq', '0')
            
            if not recording_id or recording_id not in recording_sessions:
                return jsonify({"error": "unknown recording_id"}), 404
            
            # Get the raw bytes from the request
            chunk_data = request.get_data()
            if not chunk_data:
                return jsonify({"error": "no data"}), 400
            
            # Store chunk in memory (ordered by sequence)
            session = recording_sessions[recording_id]
            session['chunks'].append((int(seq), chunk_data))
            
            return jsonify({"ok": True})
            
        except Exception as e:
            print(f"❌ Error storing chunk: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/record/stop", methods=["POST"])
    def record_stop():
        if crowd_interface._shutting_down:
            return jsonify({"status": "shutting_down"}), 503
        
        if not crowd_interface.record_demo_videos:
            return jsonify({"error": "Demo video recording is not enabled"}), 400
        
        try:
            data = request.get_json() or {}
            recording_id = data.get('recording_id')
            
            if not recording_id or recording_id not in recording_sessions:
                return jsonify({"error": "unknown recording_id"}), 404
            
            session = recording_sessions[recording_id]
            
            # Sort chunks by sequence number
            chunks = sorted(session['chunks'], key=lambda x: x[0])
            
            if not chunks:
                recording_sessions.pop(recording_id, None)
                return jsonify({"error": "no chunks received"}), 400
            
            # Combine all chunks into a single video file
            try:
                # Get next filename using the counter system
                ext = session['ext']
                filename, index = crowd_interface._next_video_filename(ext)
                file_path = crowd_interface._demo_videos_dir / filename
                
                # Write all chunks to the file
                with open(file_path, 'wb') as f:
                    for seq, chunk_data in chunks:
                        f.write(chunk_data)
                
                # Clean up session
                recording_sessions.pop(recording_id, None)
                
                # Optionally upload to blob storage (if configured)
                public_url = crowd_interface._maybe_upload_to_blob(str(file_path))
                
                return jsonify({
                    "ok": True,
                    "filename": filename,
                    "path": str(file_path),
                    "save_dir_rel": crowd_interface._rel_path_from_repo(file_path.parent),
                    "public_url": public_url,
                    "index": index
                })
                
            except Exception as e:
                print(f"❌ Error finalizing recording: {e}")
                recording_sessions.pop(recording_id, None)
                return jsonify({"error": f"failed to save: {e}"}), 500
            
        except Exception as e:
            print(f"❌ Error stopping recording: {e}")
            return jsonify({"error": str(e)}), 500
    
    return app