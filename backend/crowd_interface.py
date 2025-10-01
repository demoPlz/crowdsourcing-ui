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

from flask import Flask, jsonify
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
    "front":       4,   # change indices / paths as needed
    "left":        12,
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
                 required_responses_per_important_state=REQUIRED_RESPONSES_PER_IMPORTANT_STATE,
                 autofill_important_states: bool = False,
                 num_autofill_actions: int | None = None,
                 use_vlm_prompt: bool = False,
                 # --- NEW: saving important-state cam_main frames ---
                 save_maincam_sequence: bool = False,
                 prompt_sequence_dir: str | None = None,
                 prompt_sequence_clear: bool = False,
                 # NEW: used ONLY for prompt substitution and demo assets
                 prompt_task_name: str | None = None,
                 # --- NEW: demo video recording ---
                 record_demo_videos: bool = False,
                 demo_videos_dir: str | None = None,
                 demo_videos_clear: bool = True):

        # --- UI prompt mode (simple vs VLM) ---
        self.use_vlm_prompt = bool(use_vlm_prompt or int(os.getenv("USE_VLM_PROMPT", "0")))
        self._vlm_enabled = False  # becomes True only if VLM is requested AND configured
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
        self.required_responses_per_important_state = required_responses_per_important_state
        self.autofill_important_states = bool(autofill_important_states)
        # If not specified, default to "complete on first submission"
        if num_autofill_actions is None:
            self.num_autofill_actions = self.required_responses_per_important_state
        else:
            self.num_autofill_actions = int(num_autofill_actions)
        # Clamp to [1, required_responses_per_important_state]
        self.num_autofill_actions = max(1, min(self.num_autofill_actions,
                                               self.required_responses_per_important_state))
        
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
        # Task used for UI fallback and dataset frames → always cfg.single_task (set in init_dataset)
        self.task = None
        # Task name used for prompt placeholder substitution and demo images (from --task-name)
        self.prompt_task_name = (prompt_task_name or None)

        # Background capture state
        self._cap_threads: dict[str, Thread] = {}
        self._cap_running: bool = False
        self._latest_raw: dict[str, np.ndarray] = {}
        self._latest_ts: dict[str, float] = {}
        self._latest_proc: dict[str, np.ndarray] = {}
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

        # segmentation worker
        self._seg_queue = queue.Queue()
        self._seg_worker_thread = None
        self._seg_worker_running = False
        self._start_segmentation_worker()

        # --- VLM worker wiring (only if enabled) ---
        self._vlm_queue = queue.Queue(maxsize=_safe_int(os.getenv("VLM_QUEUE_SIZE", "8"), 8))
        self._vlm_worker_running = False
        self._vlm_worker_thread = None
        self._aoai_client = None
        self._aoai_deployment = None
        if self.use_vlm_prompt:
            # Try to verify creds early; if unavailable, fall back to simple prompts.
            if self._ensure_azure_openai_client() is not None:
                self._vlm_enabled = True
                self._start_vlm_worker()
            else:
                self._vlm_enabled = False
                print("⚠️ VLM prompts requested but Azure OpenAI is not configured; "
                      "falling back to simple task prompts.")
        else:
            print("🧪 VLM prompts disabled (use_vlm_prompt=False).")

        # NEW: prepare log dir + flags
        self._vlm_log_dir = Path(os.getenv("VLM_LOG_DIR", "output/vlm_logs")).resolve()
        try:
            self._vlm_log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self._vlm_context_done = False
        self._vlm_context_text = None

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

        # --- VLM text cache (state-specific): episode_id -> { state_id -> text }
        self._vlm_text_by_episode: dict[str, dict[int, str]] = {}

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
        self._prompt_seq_dir = Path(prompt_sequence_dir or "prompts/demo/drawer").resolve()
        self._prompt_seq_lock = Lock()
        self._prompt_seq_index = 1
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

    @staticmethod
    def _guess_ext_from_mime_or_name(mime: str | None, name: str | None, fallback: str = ".webm") -> str:
        # Conservative mapping—extend if you need more types
        mime = (mime or "").lower()
        name = (name or "")
        if "webm" in mime or name.endswith(".webm"):
            return ".webm"
        if "mp4" in mime or name.endswith(".mp4") or "mpeg4" in mime:
            return ".mp4"
        if "ogg" in mime or name.endswith(".ogv") or name.endswith(".ogg"):
            return ".ogv"
        return fallback

    def _next_video_filename(self, ext: str) -> tuple[str, int]:
        """Return ('{index}{ext}', index) and atomically increment the counter."""
        if not ext.startswith("."):
            ext = "." + ext
        with self._video_index_lock:
            idx = self._video_index
            self._video_index += 1
        return f"{idx}{ext}", idx

    def get_demo_video_config(self) -> dict:
        """
        Small, stable contract the frontend can consume.
        - enabled: whether UI should show the recorder
        - task_name: sanitized task name if provided
        - save_dir_abs / save_dir_rel: where uploads end up (for user display)
        - upload_url: endpoint the UI should POST to (multipart 'video' + optional 'metadata')
        - preferred_extension/mime: gentle hint for MediaRecorder/export
        """
        cfg = {
            "enabled": bool(self.record_demo_videos),
            "task_name": (self._task_name() or "default"),
            "save_dir_abs": None,
            "save_dir_rel": None,
            "upload_url": "/api/upload-demo-video" if self.record_demo_videos else None,
            "preferred_extension": "webm",
            "preferred_mime": "video/webm",
            # NEW: front-end hints to avoid getDisplayMedia (no Chrome picker, no screen recording)
            "suggest_canvas_capture": True,
            "filename_pattern": "{index}.{ext}",
            "sequence_start_index": 1,
            "reset_numbering_each_run": True,
            "accept_mimes": ["video/webm", "video/mp4", "video/ogg"]
        }
        if self.record_demo_videos and self._demo_videos_dir:
            cfg["save_dir_abs"] = str(self._demo_videos_dir)
            cfg["save_dir_rel"] = self._rel_path_from_repo(self._demo_videos_dir)
        return cfg

    def begin_shutdown(self):
        """Fence off new work immediately; endpoints will early-return."""
        self._shutting_down = True
        try:
            self._stop_obs_stream_worker()
        except Exception:
            pass
        try:
            self._stop_segmentation_worker()
        except Exception:
            pass
        try:
            self._stop_vlm_worker()
        except Exception:
            pass
        # Flush any remaining buffered states before shutdown
        self._flush_remaining_buffered_states()

        # Cancel any outstanding deferred finalizations
        with self.state_lock:
            for ep, t in list(self._episode_finalize_timers.items()):
                try: t.cancel()
                except Exception: pass
            self._episode_finalize_timers.clear()
    
    def _flush_remaining_buffered_states(self):
        """Flush any states remaining in the buffer, useful during shutdown"""
        if not self.dataset:
            return
            
        for episode_id, buffered_states in self.completed_states_buffer_by_episode.items():
            if buffered_states:
                print(f"🚿 Flushing {len(buffered_states)} remaining buffered states for episode {episode_id}")
                # Sort states by state_id to ensure chronological order
                for sid in sorted(buffered_states.keys()):
                    entry = buffered_states[sid]
                    if isinstance(entry, dict) and entry.get("_manifest"):
                        obs = self._load_obs_from_disk(entry.get("obs_path"))
                        frame = {**obs, "action": entry["action"], "task": entry["task"]}
                        self.dataset.add_frame(frame)
                        self._delete_obs_from_disk(entry.get("obs_path"))
                    else:
                        self.dataset.add_frame(entry)
                # Save the episode
                self.dataset.save_episode()
                # Purge any leftover cache files for this episode
                try:
                    self._purge_episode_cache(episode_id)
                except Exception:
                    pass
        
        # Clear the buffer
        self.completed_states_buffer_by_episode.clear()
        print("🧹 All buffered states flushed during shutdown")

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
                
                # ---- Check completed states in this episode (recover from completed buffer) ----
                if target_episode_id in self.completed_states_by_episode:
                    buf = self.completed_states_buffer_by_episode.get(target_episode_id, {})
                    for sid, cinfo in self.completed_states_by_episode[target_episode_id].items():
                        if sid < important_state_id and cinfo.get("is_important", False):
                            m = buf.get(sid)
                            if isinstance(m, dict) and "action" in m:
                                act = m["action"]  # concatenated 1D tensor: (required_responses_per_important_state * action_dim,)
                                action_dim = len(JOINT_NAMES)
                                first = act[:action_dim]
                                if not isinstance(first, torch.Tensor):
                                    first = torch.as_tensor(first, dtype=torch.float32)
                                # Emulate pending-state structure expected later
                                previous_important_states.append((sid, {"actions": [first]}))
                
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
                        
                        # Buffer a MANIFEST entry (avoid materializing observations into RAM)
                        completed_state = {
                            "_manifest": True,
                            "obs_path": state_info.get("obs_path"),
                            "action": all_actions,
                            "task": self.task if self.task else "crowdsourced_task",
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
            traceback.print_exc()
    
    def _start_segmentation_worker(self):
        """Start the dedicated segmentation worker thread"""
        if self._seg_worker_thread is not None and self._seg_worker_thread.is_alive():
            return
        self._seg_worker_running = True
        self._seg_worker_thread = Thread(target=self._segmentation_worker, daemon=True)
        self._seg_worker_thread.start()
        print("🧵 Started segmentation worker")

    def _stop_segmentation_worker(self):
        """Stop the segmentation worker thread (idempotent)"""
        t = getattr(self, "_seg_worker_thread", None)
        if not t:
            return
        self._seg_worker_running = False
        try: 
            self._seg_queue.put_nowait(None)
        except queue.Full: 
            pass
        if t.is_alive():
            t.join(timeout=2.0)
        self._seg_worker_thread = None
        try:
            with self._seg_queue.mutex:
                self._seg_queue.queue.clear()
        except Exception:
            pass
        print("🛑 Stopped segmentation worker")

    def _enqueue_segmentation_job(self, episode_id, state_id, view_paths, joints):
        """Enqueue a segmentation job for background processing"""
        try:
            self._seg_queue.put_nowait((episode_id, state_id, view_paths, joints))
            print(f"🗂️ Queued segmentation job ep={episode_id} sid={state_id}")
        except queue.Full:
            print(f"⚠️ Segmentation queue full; skipping ep={episode_id} sid={state_id}")

    def _segmentation_worker(self):
        """Dedicated worker thread that processes segmentation tasks from the queue"""
        while self._seg_worker_running and not self._shutting_down:
            try:
                item = self._seg_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            episode_id_local, state_id_local, view_paths_local, joints_local = item
            try:
                # heavy work with NO state_lock
                seg_paths = self._segment_views_for_state(
                    episode_id_local, state_id_local, view_paths_local, joints_local
                )
            except Exception as e:
                print(f"⚠️ segmentation failed for state {state_id_local}: {e}")
                seg_paths = {}

            # write results atomically
            with self.state_lock:
                ep_states = self.pending_states_by_episode.get(episode_id_local, {})
                info = ep_states.get(state_id_local)
                if info is not None:
                    info["segmentation_paths"] = seg_paths
                    info["segmentation_ready"] = True
                    self._segmentation_paths_by_episode.setdefault(episode_id_local, {})
                    self._segmentation_paths_by_episode[episode_id_local][state_id_local] = seg_paths
                    print(f"✅ segmentation ready for important state {state_id_local} in episode {episode_id_local}")
                else:
                    # state may have completed/been removed while we were segmenting
                    print(f"ℹ️ segmentation finished but state {state_id_local} is no longer pending")
            try:
                self._seg_queue.task_done()
            except Exception:
                pass
    
    # ---------- VLM worker plumbing ----------
    def _start_vlm_worker(self):
        if self._vlm_worker_thread is not None and self._vlm_worker_thread.is_alive():
            return
        self._vlm_worker_running = True
        self._vlm_worker_thread = Thread(target=self._vlm_worker, daemon=True, name="vlm-worker")
        self._vlm_worker_thread.start()
        print("🧵 Started VLM worker")

    def _stop_vlm_worker(self):
        t = getattr(self, "_vlm_worker_thread", None)
        if not t:
            return
        self._vlm_worker_running = False
        try: self._vlm_queue.put_nowait(None)
        except Exception: pass
        if t.is_alive():
            t.join(timeout=2.0)
        self._vlm_worker_thread = None
        try:
            with self._vlm_queue.mutex:
                self._vlm_queue.queue.clear()
        except Exception:
            pass
        print("🛑 Stopped VLM worker")

    def _enqueue_vlm_job(self, episode_id: str, state_id: int, view_paths: dict[str, str]):
        """Queue a VLM job for (episode, important_state)."""
        try:
            self._vlm_queue.put_nowait((episode_id, state_id, dict(view_paths or {})))
            print(f"🗂️ Queued VLM job ep={episode_id} sid={state_id}")
        except queue.Full:
            print(f"⚠️ VLM queue full; skipping ep={episode_id} sid={state_id}")

    # ---------- Azure OpenAI client ----------
    def _ensure_azure_openai_client(self):
        """Lazy-init Azure OpenAI client from env; returns None if not configured."""
        if self._aoai_client is not None:
            return self._aoai_client
        try:
            from openai import AzureOpenAI
        except Exception:
            print("⚠️ AzureOpenAI SDK not installed. Run: pip install openai")
            return None
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-01-preview")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # your Azure *deployment name* for GPT-5
        if not endpoint or not api_key or not deployment:
            print("⚠️ Missing Azure OpenAI env (AZURE_OPENAI_ENDPOINT/API_KEY/DEPLOYMENT). VLM disabled.")
            return None
        self._aoai_client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
        self._aoai_deployment = deployment
        return self._aoai_client

    # ---------- Blob upload (optional) ----------
    def _maybe_upload_to_blob(self, local_path: str) -> str | None:
        """
        If Azure Blob creds are set and azure-storage-blob is installed,
        upload and return a public (or SAS) URL. Otherwise return None.
        """
        try:
            from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
        except Exception:
            return None
        conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container = os.getenv("AZURE_STORAGE_CONTAINER")
        if not conn or not container:
            return None
        try:
            bsvc = BlobServiceClient.from_connection_string(conn)
            cclient = bsvc.get_container_client(container)
            try: cclient.create_container()
            except Exception: pass
            name = f"episodes/{int(time.time())}_{os.path.basename(local_path)}"
            with open(local_path, "rb") as f:
                cclient.upload_blob(name, f, overwrite=True, content_type=mimetypes.guess_type(local_path)[0] or "application/octet-stream")
            # Try SAS link (read-only, short expiry)
            try:
                from datetime import datetime, timedelta
                sas = generate_blob_sas(
                    account_name=bsvc.account_name,
                    container_name=container,
                    blob_name=name,
                    account_key=bsvc.credential.account_key,  # works for connection-string auth
                    permission=BlobSasPermissions(read=True),
                    expiry=datetime.utcnow() + timedelta(hours=12),
                )
                return f"https://{bsvc.account_name}.blob.core.windows.net/{container}/{name}?{sas}"
            except Exception:
                # Fallback to anonymous (requires container public access configured)
                return f"https://{bsvc.account_name}.blob.core.windows.net/{container}/{name}"
        except Exception as e:
            print(f"⚠️ Blob upload failed: {e}")
            return None

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

    def _task_dir(self, task_name: str | None = None) -> Path | None:
        tn = (task_name or self._task_name())
        if not tn:
            return None
        return (self._prompts_root_dir() / tn).resolve()

    def _demo_images_for_task(self, task_name: str | None = None) -> list[str]:
        """Return sorted image file paths from prompts/demo/{task-name}."""
        tn = (task_name or self._task_name())
        demo_dir = (self._prompts_root_dir() / "demo" / tn).resolve()
        if not demo_dir.exists():
            return []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        imgs = [str(p) for p in sorted(demo_dir.iterdir()) if p.suffix.lower() in exts]
        return imgs

    def _load_text(self, path: Path, fallback: str = "") -> str:
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return fallback

    def _substitute_placeholders(self, template: str, task_name: str | None = None) -> str:
        """
        Replace every {x} in 'template' with contents of prompts/{task-name}/x.txt.
        Runs up to 3 passes to allow simple nested references; leaves unknown {x} intact.
        """
        if not template:
            return ""
        tn = (task_name or self._task_name())
        tdir = self._task_dir(tn)
        if tdir is None:
            return template

        pat = re.compile(r"\{([A-Za-z0-9_\-]+)\}")
        out = template
        for _ in range(3):
            changed = False
            def repl(m):
                fname = m.group(1) + ".txt"
                fpath = (tdir / fname)
                if fpath.exists():
                    try:
                        return fpath.read_text(encoding="utf-8").strip()
                    except Exception:
                        return ""
                return m.group(0)  # leave as-is if there is no file
            new_out = pat.sub(repl, out)
            if new_out != out:
                changed = True
                out = new_out
            if not changed:
                break
        return out

    def _load_prompt_with_subst(self, fname: str, fallback: str = "") -> str:
        """
        Load prompts/{fname} and apply {x} substitution using prompts/{task-name}/x.txt.
        """
        p = (self._prompts_root_dir() / fname).resolve()
        raw = self._load_text(p, fallback)
        return self._substitute_placeholders(raw, self._task_name())

    def _ensure_vlm_log_dir(self):
        if not hasattr(self, "_vlm_log_dir"):
            self._vlm_log_dir = Path(os.getenv("VLM_LOG_DIR", "output/vlm_logs")).resolve()
            try:
                self._vlm_log_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    def _vlm_log_append(self, heading: str, payload):
        """
        Append a human-friendly log entry to output/vlm_logs/vlm.log
        and also save a timestamped snapshot file for easier diffing.
        """
        self._ensure_vlm_log_dir()
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            # Append log stream
            log_path = self._vlm_log_dir / "vlm.log"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n==== {ts} :: {heading} ====\n")
                if isinstance(payload, (dict, list)):
                    f.write(json.dumps(payload, indent=2, ensure_ascii=False))
                else:
                    f.write(str(payload))
                f.write("\n")
            # Snapshot
            snap_path = self._vlm_log_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{heading}.json"
            with open(snap_path, "w", encoding="utf-8") as g:
                json.dump(payload if isinstance(payload, (dict, list)) else {"text": str(payload)},
                          g, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _run_vlm_context_once(self):
        """
        One-time context VLM call:
          - Loads prompts/context.txt, applies {x} substitution from prompts/{task-name}/x.txt
          - Reads all images under prompts/demos/{task-name}
          - Sends a single user message with the text + all images
          - Logs query and response to output/vlm_logs
        """
        if getattr(self, "_vlm_context_done", False):
            return
        if not (self.use_vlm_prompt and self._vlm_enabled):
            return

        client = self._ensure_azure_openai_client()
        if client is None:
            return

        prompt = self._load_prompt_with_subst("context.txt",
            "You are given demonstration images for {task}. Summarize the high-level plan.")
        imgs = self._demo_images_for_task(self._task_name())
        content = [{"type": "input_text", "text": prompt}] + [
            {"type": "input_image", "image_url": self._image_file_to_data_url(p)} for p in imgs
        ]
        messages = [{"role": "user", "content": content}]

        self._vlm_log_append("CONTEXT_QUERY", {
            "task": self._task_name(),
            "prompt": prompt,
            "demo_images": imgs
        })

        out_text, raw_json = None, None
        try:
            resp = client.responses.create(model=self._aoai_deployment, input=messages)
            out_text = getattr(resp, "output_text", None)
            try:
                raw_json = resp.model_dump_json(indent=2)
            except Exception:
                raw_json = str(resp)
        except Exception as e:
            # Text-only fallback
            self._vlm_log_append("CONTEXT_FALLBACK", {"error": str(e)})
            resp = client.chat.completions.create(
                model=self._aoai_deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            out_text = (resp.choices[0].message.content
                        if getattr(resp, "choices", None) else None)
            raw_json = str(resp)

        self._vlm_log_append("CONTEXT_RESPONSE", {
            "task": self._task_name(),
            "text": out_text
        })

        # Persist a fuller record for the run
        self._ensure_vlm_log_dir()
        out_path = self._vlm_log_dir / f"{int(time.time())}_context_response.json"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "task": self._task_name(),
                    "requested_at": time.time(),
                    "prompt": prompt,
                    "demo_images": imgs,
                    "text": out_text,
                    "raw": raw_json,
                }, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        self._vlm_context_text = (out_text or "").strip()
        self._vlm_context_done = True

    # --- NEW: helpers for important-state image sequence ---
    def _compute_next_prompt_seq_index(self) -> int:
        """
        Scan the target directory and return next numeric index (1-based).
        Accepts files like 000001.jpg / 42.png / 7.jpeg, ignoring non-numeric stems.
        """
        try:
            nums = []
            for p in self._prompt_seq_dir.iterdir():
                if not p.is_file():
                    continue
                m = re.match(r"^(\d+)$", p.stem)
                if m:
                    nums.append(int(m.group(1)))
            return (max(nums) + 1) if nums else 1
        except Exception:
            return 1

    def _save_important_maincam_frame(self, episode_id: str, state_id: int, obs_path: str | None):
        """
        If enabled, load the state's obs from disk, extract cam_main, and save as
        an ordered JPEG: <dir>/<zero-padded-index>.jpg
        """
        if not self.save_maincam_sequence:
            return
        try:
            obs = self._load_obs_from_disk(obs_path)
            img = self._load_main_cam_from_obs(obs)
            if img is None:
                print(f"⚠️ No cam_main image for IMPORTANT state {state_id} (ep {episode_id})")
                return
            # Thread-safe index assignment
            with self._prompt_seq_lock:
                idx = self._prompt_seq_index
                self._prompt_seq_index += 1
            fname = f"{idx:06d}.jpg"
            out_path = self._prompt_seq_dir / fname
            # Write JPEG with current quality setting
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            ok = cv2.imwrite(str(out_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self._jpeg_quality)])
            if ok:
                print(f"🖼️  Saved IMPORTANT cam_main → {out_path}")
            else:
                print(f"⚠️ Failed to write {out_path}")
        except Exception as e:
            print(f"⚠️ Error saving IMPORTANT cam_main frame: {e}")

    def _build_episode_maincam_video(self, episode_id: str, up_to_state_id: int) -> str | None:
        """
        Assemble MP4 from cam_main frames for states in [start .. up_to_state_id] of `episode_id`.
        Returns output file path or None if no frames found.
        """
        # Collect (sid, obs_path) from both pending and completed buffers
        paths: list[tuple[int, str]] = []
        with self.state_lock:
            pend = self.pending_states_by_episode.get(episode_id, {})
            for sid, info in pend.items():
                if sid <= up_to_state_id and info.get("obs_path"):
                    paths.append((sid, info["obs_path"]))
            buf = self.completed_states_buffer_by_episode.get(episode_id, {})
            for sid, manifest in buf.items():
                if sid <= up_to_state_id and isinstance(manifest, dict) and manifest.get("obs_path"):
                    paths.append((sid, manifest["obs_path"]))
        if not paths:
            print(f"⚠️ No obs paths found to build video ep={episode_id} up_to={up_to_state_id}")
            return None

        paths.sort(key=lambda x: x[0])
        frames: list[np.ndarray] = []
        for sid, p in paths:
            obs = self._load_obs_from_disk(p)
            img = self._load_main_cam_from_obs(obs)
            if img is not None:
                frames.append(img)
        if not frames:
            print(f"⚠️ No cam_main frames found ep={episode_id} up_to={up_to_state_id}")
            return None

        H, W = frames[0].shape[:2]
        # fps: prefer dataset metadata; fall back to 8
        try:
            fps = float(getattr(self.dataset, "fps", None) or getattr(self.dataset.meta, "fps", None) or 8.0)
        except Exception:
            fps = 8.0

        out_dir = self._episode_cache_dir(episode_id)
        out_path = out_dir / f"{up_to_state_id:06d}_maincam.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
        if not writer.isOpened():
            print("⚠️ VideoWriter failed to open; trying MJPG AVI fallback")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            out_path = out_dir / f"{up_to_state_id:06d}_maincam.avi"
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

        for rgb in frames:
            if rgb.shape[:2] != (H, W):
                rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()
        print(f"🎞️ Wrote episode video → {out_path}")
        return str(out_path)

    def _gather_important_maincam_sequence(self, episode_id: str, up_to_state_id: int) -> tuple[list[str], list[int]]:
        """
        Return (image_data_urls, state_ids) for IMPORTANT states in this episode
        with state_id <= up_to_state_id, in ascending order of state_id.
        Each image is a data URL (data:image/jpeg;base64,...) produced from
        observation.images.cam_main (or the fallbacks handled by _load_main_cam_from_obs).
        """
        # Collect (sid -> obs_path) for important states from both pending and completed buffers
        paths_by_sid: dict[int, str] = {}

        with self.state_lock:
            # Completed states metadata tells us which were important; the obs paths live in the buffer
            completed_meta = self.completed_states_by_episode.get(episode_id, {})
            completed_buf  = self.completed_states_buffer_by_episode.get(episode_id, {})

            for sid, meta in completed_meta.items():
                if sid <= up_to_state_id and meta.get("is_important", False):
                    man = completed_buf.get(sid)
                    if isinstance(man, dict) and man.get("obs_path"):
                        paths_by_sid[sid] = man["obs_path"]

            # Pending states: read directly
            pending = self.pending_states_by_episode.get(episode_id, {})
            for sid, info in pending.items():
                if sid <= up_to_state_id and info.get("important", False):
                    p = info.get("obs_path")
                    if p:
                        paths_by_sid[sid] = p

        if not paths_by_sid:
            return [], []

        # Load frames in chronological order
        seq_urls: list[str] = []
        seq_ids:  list[int] = []
        for sid in sorted(paths_by_sid.keys()):
            obs = self._load_obs_from_disk(paths_by_sid[sid])
            img = self._load_main_cam_from_obs(obs)
            if img is None:
                continue
            # Encode to data URL (JPEG, quality = self._jpeg_quality)
            seq_urls.append(self._encode_jpeg_base64(img))
            seq_ids.append(sid)

        return seq_urls, seq_ids

    # ---------- VLM prompt helpers ----------
    def _load_prompt_text(self, filename: str, fallback: str = "") -> str:
        """
        Load prompt text from prompts/{filename}.
        Returns fallback text if file doesn't exist or can't be read.
        """
        try:
            base_dir = Path(__file__).resolve().parent
            prompt_path = (base_dir / ".." / "prompts" / filename).resolve()
            with open(prompt_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                return content if content else fallback
        except Exception as e:
            print(f"⚠️ Failed to load prompt from {filename}: {e}")
            return fallback

    def _get_episode_history_prompt(self) -> str:
        """History prompt (applies {x} → prompts/{task-name}/x.txt)."""
        return self._load_prompt_with_subst(
            "history.txt",
            "These are the important-state cam_main frames in chronological order for {task}."
        )

    def _get_current_state_prompt(self) -> str:
        """Current-state prompt (applies {x} → prompts/{task-name}/x.txt)."""
        return self._load_prompt_with_subst(
            "current.txt",
            "These are four synchronized views related to {task}."
        )

    # ---------- VLM worker core ----------
    def _vlm_worker(self):
        """
        For each queued (episode_id, state_id, view_paths):
          1) Gather chronological sequence of important-state cam_main frames up to current state
          2) Prepare four views from that state
          3) Call Azure OpenAI Responses API (two user messages)
          4) Save raw response JSON in the episode cache
        """
        while self._vlm_worker_running and not self._shutting_down:
            try:
                item = self._vlm_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            episode_id, state_id, view_paths = item
            try:
                # 1) Gather important-state keyframes (cam_main) for this episode up to current important state
                seq_urls, seq_ids = self._gather_important_maincam_sequence(episode_id, state_id)
                if not seq_urls:
                    print(f"⛔ VLM: no important-state cam_main sequence; skipping ep={episode_id} sid={state_id}")
                    continue
                print(f"🖼️ VLM: using {len(seq_urls)} important-state cam_main frames (sids={seq_ids})")

                # 2) Collect four views (front/left/right/perspective in this order)
                ordered = ["front", "left", "right", "perspective"]
                img_urls = []
                for name in ordered:
                    p = view_paths.get(name)
                    if p and os.path.exists(p):
                        img_urls.append(self._image_file_to_data_url(p))
                if not img_urls:
                    print("⚠️ VLM: no per-state view images found; proceeding with maincam sequence only")

                # Build content blocks
                episode_history_text = self._get_episode_history_prompt()
                current_state_text = self._get_current_state_prompt()

                # 4) Call Azure OpenAI (Responses API)
                client = self._ensure_azure_openai_client()
                if client is None:
                    print("⚠️ Azure OpenAI client not configured; skipping VLM call.")
                    continue

                print(f"\n🤖 Calling Azure OpenAI ({self._aoai_deployment}) for ep={episode_id} sid={state_id} ...")

                # ---------- inside _vlm_worker, replace the single-call block with this ----------

                # Build two prompts with substitution
                episode_history_text = self._get_episode_history_prompt()
                current_state_text   = self._get_current_state_prompt()

                # (1) HISTORY CALL: maincam sequence
                history_messages = [{
                    "role": "user",
                    "content": [{"type": "input_text", "text": episode_history_text}] +
                               [{"type": "input_image", "image_url": u} for u in seq_urls]
                }]
                self._vlm_log_append("STATE_HISTORY_QUERY", {
                    "episode_id": episode_id, "state_id": state_id,
                    "prompt": episode_history_text,
                    "sequence_state_ids": seq_ids,
                    "num_images": len(seq_urls)
                })
                hist_text, hist_raw = None, None
                try:
                    resp_h = client.responses.create(model=self._aoai_deployment, input=history_messages)
                    hist_text = getattr(resp_h, "output_text", None)
                    try:
                        hist_raw = resp_h.model_dump_json(indent=2)
                    except Exception:
                        hist_raw = str(resp_h)
                except Exception as e:
                    self._vlm_log_append("STATE_HISTORY_FALLBACK", {"episode_id": episode_id, "state_id": state_id, "error": str(e)})
                    # text-only fallback
                    resp_h = client.chat.completions.create(
                        model=self._aoai_deployment,
                        messages=[{"role": "user", "content": episode_history_text}],
                        temperature=0.0
                    )
                    hist_text = (resp_h.choices[0].message.content
                                 if getattr(resp_h, "choices", None) else None)
                    hist_raw = str(resp_h)
                self._vlm_log_append("STATE_HISTORY_RESPONSE", {
                    "episode_id": episode_id, "state_id": state_id,
                    "text": hist_text
                })

                # (2) CURRENT CALL: per-state multi-view (if any are present)
                current_messages = [{
                    "role": "user",
                    "content": [{"type": "input_text", "text": current_state_text}] +
                               [{"type": "input_image", "image_url": u} for u in img_urls]
                }]
                self._vlm_log_append("STATE_CURRENT_QUERY", {
                    "episode_id": episode_id, "state_id": state_id,
                    "prompt": current_state_text,
                    "views": list(view_paths.keys()),
                    "num_images": len(img_urls)
                })
                curr_text, curr_raw = None, None
                try:
                    resp_c = client.responses.create(model=self._aoai_deployment, input=current_messages)
                    curr_text = getattr(resp_c, "output_text", None)
                    try:
                        curr_raw = resp_c.model_dump_json(indent=2)
                    except Exception:
                        curr_raw = str(resp_c)
                except Exception as e:
                    self._vlm_log_append("STATE_CURRENT_FALLBACK", {"episode_id": episode_id, "state_id": state_id, "error": str(e)})
                    # text-only fallback
                    resp_c = client.chat.completions.create(
                        model=self._aoai_deployment,
                        messages=[{"role": "user", "content": current_state_text}],
                        temperature=0.0
                    )
                    curr_text = (resp_c.choices[0].message.content
                                 if getattr(resp_c, "choices", None) else None)
                    curr_raw = str(resp_c)
                self._vlm_log_append("STATE_CURRENT_RESPONSE", {
                    "episode_id": episode_id, "state_id": state_id,
                    "text": curr_text
                })

                # Combine for UI/storage; mark ready
                combined_text = "\n\n---\n".join(
                    [t.strip() for t in [(hist_text or ""), (curr_text or "")] if t and t.strip()]
                )

                # Persist the VLM result (now includes both calls)
                out_dir = self._episode_cache_dir(episode_id)
                out_file = out_dir / f"{state_id:06d}_vlm_response.json"
                payload = {
                    "episode_id": episode_id,
                    "state_id": state_id,
                    "requested_at": time.time(),
                    "maincam_sequence_state_ids": seq_ids,
                    "maincam_sequence_count": len(seq_ids),
                    "view_paths": view_paths,
                    "messages_history": history_messages,
                    "messages_current": current_messages,
                    "text_history": hist_text,
                    "text_current": curr_text,
                    "text_combined": combined_text,
                    "raw_history": hist_raw,
                    "raw_current": curr_raw,
                }
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                print(f"📝 Saved VLM responses (history & current) → {out_file}")

                # Attach to pending state and mark ready
                text_clean = (combined_text or "").strip()
                with self.state_lock:
                    self._vlm_text_by_episode.setdefault(episode_id, {})[state_id] = text_clean
                    ep_states = self.pending_states_by_episode.get(episode_id)
                    if ep_states and state_id in ep_states:
                        ep_states[state_id]["vlm_text"] = text_clean
                        ep_states[state_id]["vlm_ready"] = True
                        print(f"🧠 VLM (history+current) attached to state {state_id} (ep {episode_id})")
                    else:
                        print(f"ℹ️ VLM finished after state {state_id} left pending set (ep {episode_id})")

            except Exception as e:
                print(f"❌ VLM worker error: {e}")
                traceback.print_exc()
            finally:
                try: self._vlm_queue.task_done()
                except Exception: pass

    def set_events(self, events):
        """Set the events object for keyboard-like control functionality"""
        self.events = events

    def _schedule_episode_finalize_after_grace_locked(self, episode_id: str):
        """
        Schedule a deferred finalize for an empty episode.
        Assumes self.state_lock is already held.
        """
        if self._shutting_down:
            return
        # Cancel any existing timer first
        t = self._episode_finalize_timers.get(episode_id)
        if t:
            try: 
                t.cancel()
            except Exception:
                pass

        delay = self.episode_finalize_grace_s
        timer = Timer(delay, self._finalize_episode_if_still_empty, args=(episode_id,))
        timer.daemon = True
        self._episode_finalize_timers[episode_id] = timer
        timer.start()
        print(f"⏳ Scheduled deferred finalize for episode {episode_id} in {delay:.2f}s")

    def _cancel_episode_finalize_timer_locked(self, episode_id: str):
        """
        Cancel a pending finalize timer for an episode.
        Assumes self.state_lock is already held.
        """
        t = self._episode_finalize_timers.pop(episode_id, None)
        if t:
            try:
                t.cancel()
                print(f"↩️  Canceled deferred finalize for episode {episode_id}")
            except Exception:
                pass

    def _finalize_episode_if_still_empty(self, episode_id: str):
        """
        Timer callback: finalize the episode iff it remained empty through the grace period.
        Runs without holding the lock initially; acquires it to re-check and finalize.
        """
        if self._shutting_down:
            return

        with self.state_lock:
            # Clear this timer slot
            self._episode_finalize_timers.pop(episode_id, None)

            # Abort if the episode is no longer empty
            if self.pending_states_by_episode.get(episode_id):
                print(f"🛑 Finalize aborted for episode {episode_id} — new states arrived.")
                return

            # Proceed with your existing inline finalize logic (kept under lock for correctness)
            self.episodes_being_completed.add(episode_id)
            print(f"🎬 Episode {episode_id} completed (after grace). Saving episode to dataset...")

            try:
                if episode_id in self.completed_states_buffer_by_episode:
                    buffered_states = self.completed_states_buffer_by_episode[episode_id]
                    for sid in sorted(buffered_states.keys()):
                        entry = buffered_states[sid]
                        if isinstance(entry, dict) and entry.get("_manifest"):
                            obs = self._load_obs_from_disk(entry.get("obs_path"))
                            frame = {**obs, "action": entry["action"], "task": entry["task"]}
                            self.dataset.add_frame(frame)
                            self._delete_obs_from_disk(entry.get("obs_path"))
                        else:
                            self.dataset.add_frame(entry)
                    print(f"📚 Added {len(buffered_states)} states to dataset in chronological order")
                    del self.completed_states_buffer_by_episode[episode_id]
                    self._purge_episode_cache(episode_id)

                self.episodes_completed.add(episode_id)
                self.dataset.save_episode()

                # Clean up episode data structures
                if episode_id in self.pending_states_by_episode:
                    del self.pending_states_by_episode[episode_id]
                if episode_id in self.served_states_by_episode:
                    del self.served_states_by_episode[episode_id]
                if episode_id in self.completed_states_by_episode:
                    del self.completed_states_by_episode[episode_id]
                print(f"🧹 Dropped completed episode {episode_id} from monitor memory")

                # Advance serving pointer
                if self.current_serving_episode == episode_id:
                    available_episodes = [ep_id for ep_id in self.pending_states_by_episode.keys()
                                        if ep_id not in self.episodes_completed and
                                            self.pending_states_by_episode[ep_id] and ep_id is not None]
                    if available_episodes:
                        self.current_serving_episode = min(available_episodes)
                        print(f"🎬 Now serving next episode {self.current_serving_episode}")
                    else:
                        self.current_serving_episode = None
                        print("🏁 All episodes completed, no more episodes to serve")
            finally:
                self.episodes_being_completed.discard(episode_id)

    
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

        # For UI fallback and dataset writes, always use cfg.single_task
        self.task = getattr(cfg, "single_task", None)

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
        
        try:
            self._stop_obs_stream_worker()
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
        
        # --- NEW: attach segmentation masks as base64 PNGs ---
        segs = {}
        seg_paths = out.pop("segmentation_paths", None)
        if isinstance(seg_paths, dict):
            for cam, p in seg_paths.items():
                try:
                    with open(p, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("ascii")
                    segs[cam] = f"data:image/png;base64,{b64}"
                except Exception:
                    pass
        out["segments"] = segs
        
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
    
    def _global_max_pending_state_id(self) -> int | None:
        """Return newest state_id across all episodes (pending only); None if none."""
        max_id = None
        for ep_states in self.pending_states_by_episode.values():
            if not ep_states:
                continue
            k = max(ep_states.keys())
            if max_id is None or k > max_id:
                max_id = k
        return max_id
    
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
    
    # --- State Management ---
    def add_state(self, 
                  joint_positions: dict, 
                  gripper_motion: int = None, 
                  obs_dict: dict[str, torch.Tensor] = None,
                  episode_id: str = None,
                  left_carriage_external_force: float | None = None):
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
            "gripper": self._gripper_motion,
            "controls": ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper'],
            "left_carriage_external_force": (
                float(left_carriage_external_force) if left_carriage_external_force is not None else None
            )
        }
        
        # Episode-based state management - fast path to minimize latency
        state_id = self.next_state_id
        self.next_state_id += 1

        # Persist a snapshot of the camera views to disk for THIS state.
        # (Keeps index.html aligned with the exact observation that produced the state.)
        try:
            view_paths = self._persist_views_to_disk(episode_id, state_id, self._snapshot_latest_views())
        except Exception: view_paths = {}

        # Non-blocking: enqueue obs camera previews for background JPEG encoding
        try:
            if obs_dict is not None:
                self._push_obs_view("obs_main",  obs_dict.get("observation.images.cam_main"))
                self._push_obs_view("obs_wrist", obs_dict.get("observation.images.cam_wrist"))
        except Exception:
            pass
        
        # Deep copy obs_dict tensors (necessary for correctness) and spill to disk
        obs_dict_deep_copy = {}
        for key, value in obs_dict.items():
            if isinstance(value, torch.Tensor):
                obs_dict_deep_copy[key] = value.clone().detach()
            else:
                obs_dict_deep_copy[key] = value
        # Spill heavy observations to disk now and drop from RAM
        obs_path = self._persist_obs_to_disk(episode_id, state_id, obs_dict_deep_copy)
        # Explicitly release the deep copy to encourage GC
        del obs_dict_deep_copy

        state_info = {
            "state": frontend_state.copy(),
            "observations": None,           # kept on disk only
            "obs_path": obs_path,           # pointer to disk cache
            "view_paths": view_paths,
            "actions": [],
            "responses_received": 0,
            "timestamp": time.time(),
            "served_during_stationary": None,
            "important": False,
            "episode_id": episode_id,
            # NEW: segmentation bookkeeping
            "segmentation_ready": True,   # normal states don't block serving
            "segmentation_paths": {}      # view_name -> png path (filled if marked important)
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

            self._cancel_episode_finalize_timer_locked(episode_id)
        
        print(f"🟢 State {state_id} added to episode {episode_id}. Pending: {len(self.pending_states_by_episode.get(episode_id, {}))}")

    def set_last_state_to_important(self):
        """Mark the most recent state as important (across all episodes)."""
        with self.state_lock:
            latest_state_id = None
            latest_episode_id = None
            latest_timestamp = -1.0
            for episode_id, episode_states in self.pending_states_by_episode.items():
                if not episode_states:
                    continue
                for sid, sinfo in episode_states.items():
                    ts = sinfo.get("timestamp", 0.0)
                    if ts > latest_timestamp:
                        latest_timestamp = ts
                        latest_state_id = sid
                        latest_episode_id = episode_id

            if latest_state_id is None or latest_episode_id is None:
                print("⚠️  No pending states to mark as important")
                return

            info = self.pending_states_by_episode[latest_episode_id][latest_state_id]
            if 'important' in info and info['important']:
                return
            info["important"] = True
            info["segmentation_ready"] = False
            if self._vlm_enabled:
                info["vlm_text"] = None   # will be filled by VLM worker once ready
                info["vlm_ready"] = False
            else:
                # When VLM is disabled, treat as 'ready' so no gating occurs anywhere.
                info["vlm_text"] = None
                info["vlm_ready"] = True
            print(f"🔴 Marked state {latest_state_id} in episode {latest_episode_id} as important")

            # always queue auto-labeling
            self.auto_label_previous_states(latest_state_id)

            # snapshot inputs for background worker
            episode_id_local = latest_episode_id
            state_id_local = latest_state_id
            view_paths_local = dict(info.get("view_paths", {}))
            joints_local = dict(info["state"].get("joint_positions", {}))
            obs_path_local = info.get("obs_path")

            if self._vlm_enabled:
                self._enqueue_vlm_job(episode_id_local, state_id_local, view_paths_local)

            st_obj = info["state"]
            # Do not return early — we want to save the important-state image regardless.
            segmentation_needed = True
            if not self._gripper_is_grasped(st_obj):
                info["segmentation_paths"] = {}
                info["segmentation_ready"] = True
                fv = self._extract_grip_force_N(st_obj)
                print(f"⛔️ Skipping segmentation for state {latest_state_id} (ep {latest_episode_id}) — not grasped (force={fv!r})")
                segmentation_needed = False

        # enqueue heavy work after releasing the lock
        # NEW: save cam_main frame for this IMPORTANT state (ordered sequence)
        self._save_important_maincam_frame(episode_id_local, state_id_local, obs_path_local)
        if segmentation_needed:
            self._enqueue_segmentation_job(episode_id_local, state_id_local, view_paths_local, joints_local)

    def clear_episode_data(self, episode_id: str):
        """Clear all episode-related data for a specific episode (used when rerecording)"""
        with self.state_lock:

            self._cancel_episode_finalize_timer_locked(episode_id)
            
            # Clear pending states
            if episode_id in self.pending_states_by_episode:
                del self.pending_states_by_episode[episode_id]
                print(f"🧹 Cleared pending states for episode {episode_id}")
            
            # Clear completed states
            if episode_id in self.completed_states_by_episode:
                del self.completed_states_by_episode[episode_id]
                print(f"🧹 Cleared completed states for episode {episode_id}")
            
            # Clear completed states buffer
            if episode_id in self.completed_states_buffer_by_episode:
                del self.completed_states_buffer_by_episode[episode_id]
                print(f"🧹 Cleared completed states buffer for episode {episode_id}")
            
            # Clear served states
            if episode_id in self.served_states_by_episode:
                del self.served_states_by_episode[episode_id]
                print(f"🧹 Cleared served states for episode {episode_id}")
            
            # Remove from completed and being completed sets
            self.episodes_completed.discard(episode_id)
            self.episodes_being_completed.discard(episode_id)
            
            # Clear the episode cache directory
            self._purge_episode_cache(episode_id)

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
            
            # ---- Strict VLM gating for important states ----
            # Do not serve an important state until its VLM text has been generated.
            if (self._vlm_enabled
                and state_info.get("important", False)
                and not state_info.get("vlm_ready", False)):
                best_pref = None  # preferred: non-important, or important with VLM+segmentation ready
                best_any  = None  # fallback: important with VLM ready (segmentation may still be pending)

                for ep_id, ep_states in self.pending_states_by_episode.items():
                    for sid, si in ep_states.items():
                        # STRICT rule: skip important states whose VLM text isn't ready
                        if si.get("important", False) and not si.get("vlm_ready", False):
                            continue

                        ts = si.get("timestamp", 0.0)
                        candidate = (ts, sid, ep_id, si)

                        # Prefer to avoid unsegmented important states (soft preference)
                        if si.get("important", False) and not si.get("segmentation_ready", False):
                            # lower priority bucket
                            if best_any is None or ts > best_any[0]:
                                best_any = candidate
                        else:
                            # preferred bucket (non-important OR important+segmented)
                            if best_pref is None or ts > best_pref[0]:
                                best_pref = candidate

                chosen = best_pref or best_any
                if chosen:
                    _, latest_state_id, best_ep, best_si = chosen
                    state_info = best_si
                    episode_states = self.pending_states_by_episode[best_ep]
                else:
                    # No safe alternatives; block serving for now
                    print(f"⏸️ VLM not ready for important state {latest_state_id}; holding back serving.")
                    return {}

            # (Soft) preference to avoid unsegmented important states (kept for when VLM is ready)
            if state_info.get("important", False) and not state_info.get("segmentation_ready", False):
                best_sid = None; best_ep = None; best_ts = -1.0
                for ep_id, ep_states in self.pending_states_by_episode.items():
                    for sid, si in ep_states.items():
                        if si.get("important", False) and not si.get("segmentation_ready", False):
                            continue  # prefer others first
                        ts = si.get("timestamp", 0.0)
                        if ts > best_ts:
                            best_ts, best_sid, best_ep, best_si = ts, sid, ep_id, si
                if best_sid is not None:
                    state_info = best_si
                    latest_state_id = best_sid
                    episode_states = self.pending_states_by_episode[best_ep]
                # else: keep the original important state even though segmentation is pending
            
            # Mark state as served
            if state_info["served_during_stationary"] is None:
                state_info["served_during_stationary"] = not (self.robot_is_moving or self.is_async_collection)
            
            # Track which state was served to this session (episode-aware)
            serving_episode = latest_episode if (self.robot_is_moving is False and not self.is_async_collection) else self.current_serving_episode
            if serving_episode not in self.served_states_by_episode:
                self.served_states_by_episode[serving_episode] = {}
            self.served_states_by_episode[serving_episode][session_id] = latest_state_id

            # Snapshot an execution gate for this session so only this (fresh) state
            # can trigger immediate motion, and only if no newer state appears.
            if state_info["served_during_stationary"]:
                self._exec_gate_by_session[session_id] = {
                    "state_id": latest_state_id,
                    "episode_id": serving_episode,
                    # Snapshot of "newest existing state" at serve time; if this changes, gate fails.
                    "served_max_state_id": self._global_max_existing_state_id(),
                }

            
            # Prepare response
            state = state_info["state"].copy()
            state["state_id"] = latest_state_id
            state["episode_id"] = serving_episode
            # expose importance and VLM text for THIS state only
            state["is_important"] = bool(state_info.get("important", False))
            state["vlm_text"] = (
                state_info.get("vlm_text")
                or self._vlm_text_by_episode.get(serving_episode, {}).get(latest_state_id)
            )
            # expose pending flags so UI can show status (VLM only if enabled)
            state["segmentation_pending"] = bool(
                state_info.get("important", False)
                                                 and not state_info.get("segmentation_ready", False))
            state["vlm_pending"] = bool(
                self._vlm_enabled and state_info.get("important", False)
                                         and not state_info.get("vlm_ready", False))
            # Attach view_paths so serializer can serve the correct snapshot for this state
            vp = state_info.get("view_paths")
            if vp:
                state["view_paths"] = vp
            
            # Attach segmentation paths (if any) so _state_to_json can embed images
            segp = state_info.get("segmentation_paths")
            if segp:
                state["segmentation_paths"] = segp
            
            if self.is_async_collection:
                status = "async_collection"
            elif self.robot_is_moving:
                status = "moving"
            else:
                status = "stationary"
            print(f"🎯 Serving state {latest_state_id} from episode {serving_episode} to session {session_id} ({status})")
            return state

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

            # NEW: fetch originals once (also used for gripper override logic below)
            original_joints = state_info["state"].get("joint_positions", {})
            original_gripper = state_info["state"].get("gripper", 0)

            # NEW: If the frontend says the pose sliders were reset to their initial values,
            # treat this as *no pose movement* regardless of tiny IK/joint discrepancies.
            pose_reset_to_default = bool(response_data.get("pose_reset_to_default", False))

            # If not explicitly reset, fall back to joint-delta detection
            if pose_reset_to_default:
                has_joint_move = False
            else:
                MOVE_EPS = 1e-4  # small tolerance to avoid float noise
                has_joint_move = False
                for jn in JOINT_NAMES[:-1]:  # exclude the gripper slot
                    if jn in joint_positions and jn in original_joints:
                        sub = joint_positions[jn]
                        sub = sub[0] if isinstance(sub, (list, tuple)) and len(sub) > 0 else sub
                        orig = original_joints[jn]
                        orig = orig[0] if isinstance(orig, (list, tuple)) and len(orig) > 0 else orig
                        if abs(float(sub) - float(orig)) > MOVE_EPS:
                            has_joint_move = True
                            break

            effective_gripper = original_gripper if has_joint_move else gripper_action
            if has_joint_move and gripper_action != effective_gripper:
                print("✋ Ignoring gripper change in submission with joint movement; keeping original gripper.")

            # Ensure both execution (latest_goal) and recording see the sanitized gripper value
            response_data["gripper"] = effective_gripper            

            # Set as goal only if this is the exact state last served to THIS session,
            # and no newer state has been added since serving (no TTL).
            if state_info["responses_received"] == 0 and state_info["served_during_stationary"] is True:
                gate = self._exec_gate_by_session.get(session_id)
                current_max = self._global_max_existing_state_id()
                gate_ok = (
                    gate is not None
                    and gate.get("state_id") == state_id
                    and gate.get("episode_id") == found_episode
                    and current_max == gate.get("served_max_state_id")
                    and current_max == state_id
                    and self._active_episode_id == found_episode
                )
                if gate_ok:
                    self.latest_goal = response_data
                    # consume the gate so it can't be reused
                    self._exec_gate_by_session.pop(session_id, None)
                    print(f"🎯 Setting goal for immediate execution - first response to stationary state {state_id} (session={session_id})")
                else:
                    print(
                        "⏭️ Skipping immediate execution; state is stale or gate mismatch "
                        f"(gate={gate}, current_max={current_max})"
                    )
            
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
            
            goal_positions[-1] = 0.044 if effective_gripper > 0 else 0.0
            goal_positions = torch.tensor(goal_positions, dtype=torch.float32)
            state_info["actions"].append(goal_positions)

            print(f"🔔 Response recorded for state {state_id} in episode {found_episode} "
                  f"({state_info['responses_received']}/{required_responses})")

            # --- Progressive autofill for IMPORTANT states ---
            if state_info.get("important", False) and self.autofill_important_states:
                # Per submission, total filled should advance by `num_autofill_actions`
                # (1 real submission already counted above, plus (N-1) clones).
                remaining = max(0, required_responses - state_info["responses_received"])
                clones_to_add = max(0, min(self.num_autofill_actions - 1, remaining))
                for _ in range(clones_to_add):
                    state_info["actions"].append(goal_positions.clone())
                state_info["responses_received"] += clones_to_add
                if clones_to_add > 0:
                    print(f"🤖 Autofill: added {clones_to_add} clone(s) on state {state_id} → "
                          f"{state_info['responses_received']}/{required_responses}")
            
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

                # Manifest entry: keep only obs_path + action (do not materialize observations in RAM)
                manifest_entry = {
                    "_manifest": True,
                    "obs_path": state_info.get("obs_path"),
                    "action": all_actions,
                    "task": self.task if self.task else "crowdsourced_task",
                }
                
                # Buffer completed state for chronological ordering
                if found_episode not in self.completed_states_buffer_by_episode:
                    self.completed_states_buffer_by_episode[found_episode] = {}
                self.completed_states_buffer_by_episode[found_episode][state_id] = manifest_entry
                
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
                    self._schedule_episode_finalize_after_grace_locked(found_episode)
                    print(f"⏳ Episode {found_episode} currently empty; finalize scheduled after "
                        f"{self.episode_finalize_grace_s:.2f}s grace.")
                
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
        Returns {"left":{"x":..,"y":..,"z":..}, "right":{...}}; falls back to defaults.
        """
        try:
            p = self._calib_dir() / "manual_gripper_tips.json"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                # Minimal validation and float casting
                def _clean(side):
                    s = (data.get(side) or {})
                    return {
                        "x": float(s.get("x", 0.03)),
                        "y": float(s.get("y", 0.0)),
                        "z": float(s.get("z", 0.0)),
                    }
                return {"left": _clean("left"), "right": _clean("right")}
        except Exception as e:
            print(f"⚠️  failed to load manual_gripper_tips.json: {e}")
        # Defaults if missing
        return {"left": {"x": 0.03, "y": 0.0, "z": 0.0},
                "right":{"x": 0.03, "y": 0.0, "z": 0.0}}

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

    def _ensure_sam2(self):
        if self._sam2_model is not None and self._sam2_processor is not None:
            return
        if Sam2Model is None or Sam2Processor is None:
            raise RuntimeError("transformers Sam2Model/Sam2Processor not installed.")
        print(f"🧠 Loading SAM2 '{self._sam2_model_id}' on {self._sam2_device} ...")
        try:
            self._sam2_model = Sam2Model.from_pretrained(self._sam2_model_id).to(self._sam2_device)
        except Exception as e:
            # Fallback to a commonly-available ID
            fallback_id = "facebook/sam2-hiera-tiny"
            if self._sam2_model_id != fallback_id:
                print(f"⚠️  Failed to load {self._sam2_model_id}: {e}. Falling back to '{fallback_id}'.")
                self._sam2_model = Sam2Model.from_pretrained(fallback_id).to(self._sam2_device)
            else:
                raise
        self._sam2_model.eval()
        self._sam2_processor = Sam2Processor.from_pretrained(self._sam2_model_id)
        print("✅ SAM2 ready.")

    @staticmethod
    def _load_rgb_image_from_file(path: str) -> np.ndarray | None:
        if not path or not os.path.exists(path):
            return None
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _episode_seg_dir(self, episode_id: str) -> Path:
        d = self._episode_cache_dir(episode_id) / "segments"
        try: d.mkdir(parents=True, exist_ok=True)
        except Exception: pass
        return d

    def _save_mask_png(self, episode_id: str, state_id: int, view_name: str,
                       mask_bool: np.ndarray, img_rgb: np.ndarray) -> str:
        """
        Save a PNG whose RGB are the source view *under the mask*, and whose
        alpha is the mask (opaque where mask==True, transparent elsewhere).

        This overlays perfectly on the background view and can be translated as 1 unit.
        """
        # Ensure we have matching sizes
        ih, iw = img_rgb.shape[:2]
        mh, mw = mask_bool.shape[:2]
        if (mh, mw) != (ih, iw):
            # Safety: resize mask to image size with nearest-neighbor (no blur)
            mask_bool = cv2.resize(mask_bool.astype(np.uint8), (iw, ih),
                                   interpolation=cv2.INTER_NEAREST) > 0

        # Compose BGRA where alpha = 255 inside mask
        m = mask_bool.astype(np.uint8)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # cv2 expects BGR/BGRA

        out = np.zeros((ih, iw, 4), dtype=np.uint8)
        out[..., 0] = bgr[..., 0] * m  # B
        out[..., 1] = bgr[..., 1] * m  # G
        out[..., 2] = bgr[..., 2] * m  # R
        out[..., 3] = m * 255          # A

        out_path = self._episode_seg_dir(episode_id) / f"{state_id}_{view_name}.png"
        cv2.imwrite(str(out_path), out)
        return str(out_path)

    @staticmethod
    def _patch_dark_mean(img_rgb: np.ndarray, u: int, v: int, r: int) -> float:
        H, W, _ = img_rgb.shape
        x0 = max(0, u - r); x1 = min(W, u + r + 1)
        y0 = max(0, v - r); y1 = min(H, v + r + 1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        gray = cv2.cvtColor(img_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
        return float(gray.mean())

    @staticmethod
    def _patch_rgb_mean(img_rgb: np.ndarray, u: int, v: int, r: int) -> tuple[float, float, float]:
        """
        Mean R,G,B over a (2r+1)x(2r+1) patch centered at (u,v).
        Returns (R,G,B) in 0..255 (float32).
        """
        H, W, _ = img_rgb.shape
        x0 = max(0, u - r); x1 = min(W, u + r + 1)
        y0 = max(0, v - r); y1 = min(H, v + r + 1)
        if x1 <= x0 or y1 <= y0:
            return (0.0, 0.0, 0.0)
        patch = img_rgb[y0:y1, x0:x1]  # RGB
        mean_rgb = patch.reshape(-1, 3).mean(axis=0).astype(np.float32)
        return float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])

    def _is_flat_gray_seed(self, mean_rgb: tuple[float, float, float]) -> bool:
        """
        True if each channel is within [gray_min, gray_max] and
        max(R,G,B)-min(R,G,B) <= gray_delta (≈achromatic/neutral gray).
        """
        if self._seg_gray_delta <= 0 or self._seg_gray_min >= self._seg_gray_max:
            return False
        r, g, b = mean_rgb
        if not (self._seg_gray_min <= r <= self._seg_gray_max): return False
        if not (self._seg_gray_min <= g <= self._seg_gray_max): return False
        if not (self._seg_gray_min <= b <= self._seg_gray_max): return False
        return (max(r, g, b) - min(r, g, b)) <= self._seg_gray_delta

    def _project_world_to_pixel(self, view_name: str, Pw: np.ndarray) -> tuple[int, int] | None:
        """
        Project world point Pw (x,y,z in meters) into pixel (u,v) for 'view_name'
        using current T_three (camera pose) and Knew (intrinsics).
        """
        pose_key = f"{view_name}_pose"
        M = self._camera_poses.get(pose_key)
        model = self._camera_models.get(view_name)
        if M is None or model is None or "Knew" not in model:
            return None

        M = np.asarray(M, dtype=np.float64)  # 4x4 (Three.js world matrix)
        R_three = M[:3, :3]
        t_world = M[:3, 3]

        # world → three-camera coords (camera looks along -Z)
        Xc_three = R_three.T @ (np.asarray(Pw).reshape(3,) - t_world)

        # three-camera → OpenCV camera (+Z forward)
        Rflip = np.diag([1.0, -1.0, -1.0])
        Xc_cv = Rflip @ Xc_three
        Z = Xc_cv[2]
        if Z <= 0:
            return None

        K = np.asarray(model["Knew"], dtype=np.float64)
        u = int(round(K[0, 0] * (Xc_cv[0] / Z) + K[0, 2]))
        v = int(round(K[1, 1] * (Xc_cv[1] / Z) + K[1, 2]))
        return (u, v)

    def _sam2_segment_point(self, img_rgb: np.ndarray, u: int, v: int, multimask_output: bool = False) -> np.ndarray | None:
        self._ensure_sam2()
        try:
            inputs = self._sam2_processor(
                images=Image.fromarray(img_rgb),
                input_points=[[[[float(u), float(v)]]]],
                input_labels=[[[1]]],
                return_tensors="pt"
            ).to(self._sam2_device)

            with torch.no_grad():
                outputs = self._sam2_model(**inputs, multimask_output=multimask_output)

            masks = self._sam2_processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                inputs["original_sizes"]
            )[0]  # usually (N, H, W) but some builds emit (N,1,H,W)

            if masks is None or masks.shape[0] == 0:
                return None

            m = masks[0].detach().cpu().numpy()  # first proposal
            m = np.squeeze(m)                    # drop any singleton dims
            # Final safety: if still >2D, take the last 2 dims as (H,W)
            if m.ndim > 2:
                m = m.reshape(m.shape[-2], m.shape[-1])
            if m.ndim != 2:
                raise ValueError(f"SAM2 mask has unexpected shape {m.shape}")

            return (m >= 0.5)
        except Exception as e:
            print(f"❌ SAM2 inference failed: {e}")
            return None

    def _generate_ring_points(self, u: int, v: int, W: int, H: int,
                              base_radius_px: int, rings: int, per_ring: int) -> list[tuple[int,int]]:
        """Return [(u,v)] (center) plus concentric rings of integer pixel points within image bounds."""
        pts = [(u, v)]
        for k in range(1, rings + 1):
            r = k * base_radius_px
            for j in range(per_ring):
                theta = 2.0 * math.pi * (j / per_ring)
                uu = int(round(u + r * math.cos(theta)))
                vv = int(round(v + r * math.sin(theta)))
                if 0 <= uu < W and 0 <= vv < H:
                    pts.append((uu, vv))
        return pts

    def _filter_seed_points(self, img_rgb: np.ndarray,
                            pts: list[tuple[int,int]],
                            mode: str = "strict") -> list[tuple[int,int]]:
        """
        mode = 'off'   : keep all in-bounds points
        mode = 'loose' : only reject *very* dark seeds; ignore flat-gray
        mode = 'strict': current behavior (dark + flat-gray guards)
        """
        if mode not in ("off", "loose", "strict"):
            mode = "strict"
        out = []
        for uu, vv in pts:
            if mode == "off":
                out.append((uu, vv))
                continue
            mean_dark = self._patch_dark_mean(img_rgb, uu, vv, self._seg_dark_patch_radius)
            if mode == "loose":
                if mean_dark < 0.5 * self._seg_dark_mean_thresh:  # only reject *very* dark
                    continue
                out.append((uu, vv))
                continue
            # strict
            if mean_dark < self._seg_dark_mean_thresh:
                continue
            mean_rgb = self._patch_rgb_mean(img_rgb, uu, vv, self._seg_gray_patch_radius)
            if self._is_flat_gray_seed(mean_rgb):
                continue
            out.append((uu, vv))
        return out

    def _sam2_segment_points(self, img_rgb: np.ndarray,
                             pos_points: list[tuple[int,int]],
                             neg_points: list[tuple[int,int]] | None = None,
                             multimask_output: bool = True) -> np.ndarray | None:
        """
        One SAM2 forward with multi-point prompts.
        pos_points -> label 1, neg_points -> label 0.
        Returns a binary mask (H,W) or None.
        """
        self._ensure_sam2()
        neg_points = neg_points or []
        if not pos_points and not neg_points:
            print("⛔ SAM2: no prompt points (need ≥1 positive or negative) → skipping")
            return None
        try:
            all_pts = pos_points + neg_points
            labels  = [1] * len(pos_points) + [0] * len(neg_points)
            inputs = self._sam2_processor(
                images=Image.fromarray(img_rgb),
                input_points=[[[[float(u), float(v)] for (u, v) in all_pts]]],
                input_labels=[[[int(l) for l in labels]]],
                return_tensors="pt"
            ).to(self._sam2_device)

            with torch.no_grad():
                outputs = self._sam2_model(**inputs, multimask_output=multimask_output)

            masks = self._sam2_processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                inputs["original_sizes"]
            )[0]

            if masks is None:
                return None

            # Collect candidate masks (1..N)
            cand = []
            if masks.ndim == 2:
                cand = [masks]
            elif masks.ndim == 3:
                n = masks.shape[0]
                for i in range(n):
                    cand.append(masks[i])
            else:
                raise ValueError(f"SAM2 masks unexpected shape {masks.shape}")

            # Score: positives inside minus 2x negatives inside
            best_mask = None
            best_score = -1e18
            for m in cand:
                mb = (np.squeeze(m) >= 0.5)
                score = 0
                for (u, v) in pos_points:
                    if 0 <= v < mb.shape[0] and 0 <= u < mb.shape[1] and mb[v, u]:
                        score += 1
                for (u, v) in neg_points:
                    if 0 <= v < mb.shape[0] and 0 <= u < mb.shape[1] and mb[v, u]:
                        score -= 2
                if score > best_score:
                    best_score = score
                    best_mask = mb

            # Optional morphological closing
            k = int(self._seg_mask_close_ksize)
            if best_mask is not None and k >= 3:
                if k % 2 == 0: k += 1
                kernel = np.ones((k, k), np.uint8)
                best_mask = cv2.morphologyEx(best_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel) > 0

            return best_mask

        except Exception as e:
            print(f"❌ SAM2 multi-point inference failed: {e}")
            return None
        
    def _segment_views_for_state(self, episode_id: str, state_id: int, view_paths: dict[str, str], joint_positions: dict) -> dict[str, str]:
        """
        Compute gripper center (URDF FK) → project to each calibrated view → run SAM2 if not dark.
        Returns {view_name: mask_png_path}
        """
        out_paths: dict[str, str] = {}
        # find state & its per-view files
        if not view_paths:
            print(f"⚠️ segmentation: no view_paths for state {state_id}")
            return out_paths

        # 1) FK → world gripper center
        Pw = self._gripper_center_from_joints(joint_positions or {})
        if Pw is None:
            print(f"⚠️ segmentation: FK failed (no URDF or bad joints) for state {state_id}")
            return out_paths

        # 2) Iterate VIEWS only (skip obs_*); require current calibration
        for view_name, img_path in view_paths.items():
            if view_name not in self._camera_models:  # skip obs_main/obs_wrist, etc.
                continue

            img = self._load_rgb_image_from_file(img_path)
            if img is None:
                continue
            H, W, _ = img.shape

            uv = self._project_world_to_pixel(view_name, Pw)
            if uv is None:
                print(f"⚠️ projection failed for view '{view_name}' (state {state_id})")
                continue
            u, v = uv
            if not (0 <= u < W and 0 <= v < H):
                print(f"⚠️ seed ({u},{v}) outside image for '{view_name}'")
                continue

            # 3) multi-seed around projected center (u,v)
            if self._seg_use_multi_seed:
                # Decide center role
                center_is_dark = (self._patch_dark_mean(img, u, v, self._seg_dark_patch_radius) < self._seg_dark_mean_thresh)
                center_is_gray = self._is_flat_gray_seed(self._patch_rgb_mean(img, u, v, self._seg_gray_patch_radius))
                use_center_as_neg = self._seg_center_negative and (center_is_dark or center_is_gray)

                # Build positive ring points outside the gripper
                base_r = max(self._seg_seed_radius_px, self._seg_seed_min_radius_px)
                ring_pts = self._generate_ring_points(
                    u, v, W, H,
                    base_radius_px=base_r,
                    rings=max(1, self._seg_seed_rings),
                    per_ring=max(6, self._seg_seed_per_ring)
                )

                # Loosen filtering on ring points so object pixels survive
                pos_pts = self._filter_seed_points(img, ring_pts, mode=self._seg_ring_filter_mode)

                # Ensure at least one positive; escalate radius if necessary
                tries = 0
                while len(pos_pts) < max(1, self._seg_min_valid_seeds) and tries < 2:
                    base_r = int(round(base_r * 1.6))
                    ring_pts = self._generate_ring_points(u, v, W, H, base_r, rings=1, per_ring=max(8, self._seg_seed_per_ring))
                    pos_pts = self._filter_seed_points(img, ring_pts, mode=self._seg_ring_filter_mode)
                    tries += 1

                # Fallback: if still empty, accept ring points unfiltered
                if len(pos_pts) == 0:
                    pos_pts = ring_pts[:]  # in-bounds by construction

                # Negatives: center (to reject the black gripper) plus optional outer ring
                neg_pts = []
                if use_center_as_neg:
                    neg_pts.append((u, v))
                if self._seg_use_neg_ring:
                    outer_r = int(round(base_r * self._seg_neg_ring_scale))
                    neg_ring = self._generate_ring_points(u, v, W, H, base_radius_px=outer_r, rings=1,
                                                          per_ring=max(8, self._seg_seed_per_ring))
                    neg_pts.extend(neg_ring)

                # 4) SAM2 with multi-point prompt; let it return the best candidate
                mask = self._sam2_segment_points(img, pos_pts, neg_pts, multimask_output=self._seg_multimask)

                # Debugging (optional)
                if mask is None:
                    print(f"⛔ multi-seed produced no mask: pos={len(pos_pts)} neg={len(neg_pts)} (center_neg={use_center_as_neg})")

            else:
                # Legacy single-seed path (unchanged)
                mean_val = self._patch_dark_mean(img, u, v, self._seg_dark_patch_radius)
                if mean_val < self._seg_dark_mean_thresh:
                    print(f"⛔ view '{view_name}': dark patch at seed (mean={mean_val:.1f}) → skip")
                    continue
                mask = self._sam2_segment_point(img, u, v, multimask_output=False)
            if mask is None:
                print(f"❌ SAM2 mask None for '{view_name}'")
                continue

            # 5) save cut-out (RGB under mask + alpha), not a black silhouette
            out_path = self._save_mask_png(episode_id, state_id, view_name, mask, img)
            out_paths[view_name] = out_path
            print(f"🎯 Segmented '{view_name}' → {out_path}")

        return out_paths

    # ============================================================

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
        
        # Generate or retrieve session ID from request headers or IP
        session_id = request.headers.get('X-Session-ID', request.remote_addr)
        
        state = crowd_interface.get_latest_state(session_id)
        payload = crowd_interface._state_to_json(state)
        
        # Use state-specific VLM text if present for THIS state; otherwise fall back
        use_vlm = bool(getattr(crowd_interface, "use_vlm_prompt", False)
                       and getattr(crowd_interface, "_vlm_enabled", False))
        vlm_text = payload.get("vlm_text")
        if use_vlm and isinstance(vlm_text, str) and vlm_text.strip():
            payload["prompt"] = f"{vlm_text.strip()} Animate to check that there's no collision."
        else:
            payload["prompt"] = f"Task: {crowd_interface.task}. What should the arm do next?"

        # NEW: Always tell the frontend what to do with demo videos
        payload["demo_video"] = crowd_interface.get_demo_video_config()

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
        if crowd_interface._shutting_down:
            return jsonify({"status": "shutting_down"}), 503
        
        data = request.get_json(force=True, silent=True) or {}
        
        # Generate or retrieve session ID from request headers or IP
        session_id = request.headers.get('X-Session-ID', request.remote_addr)
        
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
            newest_state = newest_state_data["state"]
            monitoring_data = {
                "status": "success",
                "state_id": newest_state_id,
                "episode_id": newest_episode_id,
                "current_serving_episode": current_episode,
                "timestamp": newest_state_data["timestamp"],
                "responses_received": newest_state_data["responses_received"],
                "responses_required": (
                    crowd_interface.required_responses_per_important_state
                    if newest_state_data.get("important", False)
                    else crowd_interface.required_responses_per_state
                ),
                "is_important": newest_state_data.get("important", False),
                "views": crowd_interface._snapshot_latest_views(),  # lightweight snapshot (pre-encoded)
                "joint_positions": newest_state.get("joint_positions", {}),
                "gripper": newest_state.get("gripper", 0),
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
            
            # Get optional metadata
            metadata = {}
            if 'metadata' in request.form:
                try:
                    metadata = json.loads(request.form['metadata'])
                except:
                    pass
            
            # Force server-side sequential naming: 1.ext, 2.ext, ...
            # Decide extension from MIME or original name; default ".webm"
            ext = CrowdInterface._guess_ext_from_mime_or_name(
                getattr(file, "mimetype", None), file.filename, fallback=crowd_interface._video_ext_default
            )
            filename, index = crowd_interface._next_video_filename(ext)
            file_path = crowd_interface._demo_videos_dir / filename
            file.save(str(file_path))
            
            # Optionally save metadata
            if metadata:
                # Save metadata as {index}.json alongside the video, regardless of original filename
                metadata_path = (crowd_interface._demo_videos_dir / f"{index}.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
            
            # Optionally upload to blob storage (if configured) to produce a shareable URL
            public_url = crowd_interface._maybe_upload_to_blob(str(file_path))

            return jsonify({
                "status": "success",
                "filename": filename,
                "path": str(file_path),  # absolute on server (for logs)
                "save_dir_rel": crowd_interface._rel_path_from_repo(file_path.parent),
                "public_url": public_url,  # may be None if blob not configured
                "config": crowd_interface.get_demo_video_config(),
                "index": index
            })
            
        except Exception as e:
            print(f"❌ Error uploading demo video: {e}")
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500
    
    return app