"""
CrowdInterface - Main Backend Interface

Coordinates all subsystem managers to provide an API for crowd-sourced robot data collection.
"""

import base64
import os
import tempfile
import time
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from hardware_config import CAM_IDS, REAL_CALIB_PATHS, SIM_CALIB_PATHS
from interface_managers.calibration_manager import CalibrationManager
from interface_managers.dataset_manager import DatasetManager
from interface_managers.demo_video_manager import DemoVideoManager
from interface_managers.drawer_position_manager import DrawerPositionManager
from interface_managers.observation_stream_manager import ObservationStreamManager
from interface_managers.pose_estimation_manager import PoseEstimationManager
from interface_managers.sim_manager import SimManager
from interface_managers.state_manager import StateManager
from interface_managers.webcam_manager import WebcamManager


class CrowdInterface:
    """Main interface between frontend and backend for CAPTCHA-style crowd-sourced data collection for robot
    manipulation.

    Coordinates all subsystem managers:
    - State management (episodes, states, responses)
    - Camera/observation handling
    - Dataset operations
    - Calibration
    - Demo videos and prompts
    - Simulation rendering
    - Pose estimation

    """

    # =========================
    # Initialization
    # =========================

    def __init__(
        self,
        required_responses_per_state: int = 1,
        required_responses_per_critical_state: int = 10,
        autofill_critical_states: bool = False,
        num_autofill_actions: int | None = None,
        use_manual_prompt: bool = False,
        # --- saving critical-state cam_main frames ---
        save_maincam_sequence: bool = False,
        prompt_sequence_dir: str | None = None,
        prompt_sequence_clear: bool = False,
        # used ONLY for prompt substitution and demo assets
        task_name: str | None = None,
        # --- demo video recording ---
        record_demo_videos: bool = False,
        demo_videos_dir: str | None = None,
        demo_videos_clear: bool = False,
        # --- read-only demo video display (independent of recording) ---
        show_demo_videos: bool = False,
        show_videos_dir: str | None = None,
        # --- sim ---
        use_sim: bool = True,
        # --- objects ---
        objects: dict[str, str] | None = None,
        object_mesh_paths: dict[str, str] | None = None,
    ):

        # --- UI prompt mode (simple vs MANUAL) ---
        self.use_manual_prompt = bool(use_manual_prompt or int(os.getenv("USE_MANUAL_PROMPT", "0")))

        # --- Sim ---
        self.use_sim = use_sim

        # --- Objects ---
        self.objects = objects
        self.object_mesh_paths = object_mesh_paths

        # -------- Observation disk cache (spills heavy per-state obs to disk) --------
        # Set CROWD_OBS_CACHE to override where temporary per-state observations are stored.
        self._obs_cache_root = Path(
            os.getenv("CROWD_OBS_CACHE", os.path.join(tempfile.gettempdir(), "crowd_obs_cache"))
        )
        try:
            self._obs_cache_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

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
        self.num_autofill_actions = max(1, min(self.num_autofill_actions, self.required_responses_per_critical_state))

        # Episode-based state management (shared with StateManager via reference)
        self.pending_states_by_episode = {}  # episode_id -> {state_id -> {state: dict, responses_received: int}}
        self.completed_states_by_episode = (
            {}
        )  # episode_id -> {state_id -> {responses_received: int, completion_time: float}}
        self.completed_states_buffer_by_episode = (
            {}
        )  # episode_id -> {state_id -> completed_state_dict} - buffer for chronological add_frame
        self.served_states_by_episode = {}  # episode_id -> {session_id -> state_id}
        self.episodes_being_completed = set()  # Track episodes currently being processed for completion

        self.state_lock = Lock()  # Protects all episode-based state management

        # Task used for UI fallback and dataset frames -> always cfg.single_task (set in init_dataset)
        self.task_text = None
        # Task name used for prompt placeholder substitution and demo images (from --task-name)
        self.task_name = task_name

        # Calibration manager
        repo_root = Path(__file__).resolve().parent.parent
        self.calibration = CalibrationManager(
            use_sim=self.use_sim,
            repo_root=repo_root,
            real_calib_paths=REAL_CALIB_PATHS,
            sim_calib_paths=SIM_CALIB_PATHS,
        )

        # Demo video manager
        self.video_manager = DemoVideoManager(
            task_name=task_name,
            record_demo_videos=record_demo_videos,
            demo_videos_dir=demo_videos_dir,
            demo_videos_clear=demo_videos_clear,
            show_demo_videos=show_demo_videos,
            show_videos_dir=show_videos_dir,
            save_maincam_sequence=save_maincam_sequence,
            prompt_sequence_dir=prompt_sequence_dir,
            prompt_sequence_clear=prompt_sequence_clear,
            repo_root=repo_root,
        )

        # Webcam manager
        self.webcam_manager = WebcamManager(
            cam_ids=CAM_IDS,
            undistort_maps=self.calibration.get_undistort_maps(),
            jpeg_quality=int(os.getenv("JPEG_QUALITY", "80")),
        )

        # Observation stream manager
        self.obs_stream = ObservationStreamManager(encoder_func=self.webcam_manager.encode_jpeg_base64)

        # Sim manager
        self.sim_manager = SimManager(
            use_sim=self.use_sim,
            task_name=task_name,
            obs_cache_root=self._obs_cache_root,
            state_lock=self.state_lock,
            pending_states_by_episode=self.pending_states_by_episode,
        )

        # Debounced episode finalization
        self.episode_finalize_grace_s = 2.0

        # Precompute immutable views and camera poses to avoid per-tick allocations

        self._exec_gate_by_session: dict[str, dict] = {}

        # Pose estimation manager
        self.pose_estimator = PoseEstimationManager(
            obs_cache_root=self._obs_cache_root,
            object_mesh_paths=object_mesh_paths,
            objects=objects,
            calibration_manager=self.calibration,
            state_lock=self.state_lock,
            pending_states_by_episode=self.pending_states_by_episode,
        )

        # Drawer position manager
        self.drawer_position = DrawerPositionManager(
            calibration_manager=self.calibration,
            drawer_joint_name="Drawer_Joint",
            repo_root=repo_root,
        )

        # --- Episode save behavior: datasets are always auto-saved after finalization ---
        # Manual save is only used for demo video recording workflow
        self._episodes_pending_save: set[str] = set()

        # Dataset manager
        self.dataset_manager = DatasetManager(
            required_responses_per_critical_state=self.required_responses_per_critical_state,
            obs_cache_root=self._obs_cache_root,
        )

        # State manager (handles episode-based state lifecycle)
        self.state_manager = StateManager(
            required_responses_per_state=self.required_responses_per_state,
            required_responses_per_critical_state=self.required_responses_per_critical_state,
            autofill_critical_states=self.autofill_critical_states,
            num_autofill_actions=self.num_autofill_actions,
            use_manual_prompt=self.use_manual_prompt,
            use_sim=self.use_sim,
            task_text=self.task_text,
            obs_cache_root=self._obs_cache_root,
            state_lock=self.state_lock,
            pending_states_by_episode=self.pending_states_by_episode,
            completed_states_by_episode=self.completed_states_by_episode,
            completed_states_buffer_by_episode=self.completed_states_buffer_by_episode,
            episode_finalize_grace_s=self.episode_finalize_grace_s,
            episodes_pending_save=self._episodes_pending_save,
            obs_stream_manager=self.obs_stream,
            pose_estimation_manager=self.pose_estimator,
            drawer_position_manager=self.drawer_position,
            sim_manager=self.sim_manager,
            persist_views_callback=self._persist_views_to_disk,
            persist_obs_callback=self._persist_obs_to_disk,
            snapshot_views_callback=self.snapshot_latest_views,
            save_episode_callback=self.dataset_manager.save_episode,
        )

    # =========================
    # Camera & Observation Management
    # =========================

    def init_cameras(self):
        """Open webcams and start background capture.

        Delegates to WebcamManager.

        """
        self.webcam_manager.init_cameras()

    def snapshot_latest_views(self) -> dict[str, str]:
        """Snapshot the latest **JPEG base64 strings** for each camera.

        Includes both webcam views and observation camera previews.

        """
        # Get webcam views from manager
        out = self.webcam_manager.snapshot_latest_views()

        # Include latest observation camera previews from manager
        out.update(self.obs_stream.get_latest_obs_jpeg())

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
        out.pop("obs_path", None)  # don't expose obs cache paths

        # Prefer state-aligned snapshots if available
        views = {}
        view_paths = out.pop("view_paths", None)  # don't expose file paths to the client
        views = self._load_views_from_disk(view_paths)

        out["views"] = views
        out["camera_poses"] = self.calibration.get_camera_poses()
        out["camera_models"] = self.calibration.get_camera_models()
        out["gripper_tip_calib"] = self.calibration.get_gripper_tip_calib()

        # --- Attach example video URL (direct file URL; byte-range capable) ---
        if self.video_manager.show_demo_videos:
            # Prefer a VLM-selected clip if available and present
            video_id = state.get("video_prompt")
            chosen_url = None
            if video_id is not None:
                p, _ = self.video_manager.find_show_video_by_id(video_id)
                if p:
                    chosen_url = f"/api/show-videos/{video_id}"  # serves the exact id

            # Fallback: latest available .webm
            if not chosen_url:
                lp, lid = self.video_manager.find_latest_show_video()
                if lp and lid:
                    # Stable "latest" URL for the player; resolves dynamically on the server
                    chosen_url = "/api/show-videos/latest.webm"

            if chosen_url:
                out["example_video_url"] = chosen_url

        return out

    def encode_jpeg_base64(self, img_rgb: np.ndarray, quality: int | None = None) -> str:
        """Encode an RGB image to a base64 JPEG data URL.

        Delegates to WebcamManager.

        """
        return self.webcam_manager.encode_jpeg_base64(img_rgb, quality)

    # =========================
    # State Management (Delegated to StateManager)
    # =========================

    def add_state(
        self,
        joint_positions: dict,
        gripper_motion: int = None,
        obs_dict: dict[str, torch.Tensor] = None,
        episode_id: str = None,
        left_carriage_external_force: float | None = None,
    ):
        """Add a new state to the current episode.

        Delegates to StateManager.

        """
        return self.state_manager.add_state(
            joint_positions=joint_positions,
            gripper_motion=gripper_motion,
            obs_dict=obs_dict,
            episode_id=episode_id,
            left_carriage_external_force=left_carriage_external_force,
        )

    def set_last_state_to_critical(self):
        """Mark the last added state as critical.

        Delegates to StateManager.

        """
        return self.state_manager.set_last_state_to_critical()

    def get_latest_state(self) -> dict:
        """Get the latest pending critical state for labeling.

        Delegates to StateManager.

        """
        return self.state_manager.get_latest_state()

    def record_response(self, response_data: dict):
        """Record a user response for a state.

        Delegates to StateManager.

        """
        return self.state_manager.record_response(response_data)

    def get_pending_states_info(self) -> dict:
        """Get episode-based state information for monitoring.

        Delegates to StateManager.

        """
        return self.state_manager.get_pending_states_info()

    def set_active_episode(self, episode_id):
        """Mark which episode the robot loop is currently in.

        Delegates to StateManager.

        """
        return self.state_manager.set_active_episode(episode_id)

    def set_prompt_ready(
        self, state_info: dict, episode_id: int, state_id: int, text: str | None, video_id: int | None
    ) -> None:
        """Set prompt fields and mark state as ready.

        Delegates to StateManager.

        """
        return self.state_manager.set_prompt_ready(state_info, episode_id, state_id, text, video_id)

    def get_latest_goal(self) -> dict | None:
        """Get and clear the latest goal for robot execution.

        Delegates to StateManager.

        """
        return self.state_manager.get_latest_goal()

    # =========================
    # Reset State Management
    # =========================

    def start_reset(self, duration_s: float):
        """Start the reset countdown timer."""
        self.is_resetting = True
        self.reset_start_time = time.time()
        self.reset_duration_s = duration_s
        print(f"🔄 Starting reset countdown: {duration_s}s")

    def stop_reset(self):
        """Stop the reset countdown timer."""
        self.is_resetting = False
        self.reset_start_time = None
        self.reset_duration_s = 0

    def get_reset_countdown(self) -> float:
        """Get remaining reset time in seconds, or 0 if not resetting."""
        if not self.is_resetting or self.reset_start_time is None:
            return 0

        elapsed = time.time() - self.reset_start_time
        remaining = max(0, self.reset_duration_s - elapsed)

        # Auto-stop when countdown reaches 0
        if remaining <= 0 and self.is_resetting:
            self.stop_reset()

        return remaining

    def is_in_reset(self) -> bool:
        """Check if currently in reset state."""
        return self.is_resetting and self.get_reset_countdown() > 0

    # =========================
    # Dataset Management (Delegated to DatasetManager)
    # =========================

    def init_dataset(self, cfg, robot):
        """Initialize dataset for data collection policy training.

        Delegates to DatasetManager.

        """
        self.task_text = self.dataset_manager.init_dataset(cfg, robot)

        # Update state manager's task_text since it was None during initialization
        self.state_manager.task_text = self.task_text

    # =========================
    # Calibration Management (Delegated to CalibrationManager)
    # =========================

    def save_gripper_tip_calibration(self, calib: dict) -> str:
        """Save gripper tip calibration and return the written path.

        Delegates to CalibrationManager.

        """
        return self.calibration.save_gripper_tip_calibration(calib)

    # =========================
    # Observation Cache Management (disk persistence)
    # =========================

    def _episode_cache_dir(self, episode_id: str) -> Path:
        """Get or create cache directory for an episode."""
        d = self._obs_cache_root / str(episode_id)
        if not d.exists():
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        return d

    def _persist_obs_to_disk(self, episode_id: str, state_id: int, obs: dict) -> str | None:
        """Writes the observations dict to a single file for the state and returns the path."""
        try:
            p = self._episode_cache_dir(episode_id) / f"{state_id}.pt"
            # Tensors/ndarrays/py objects handled by torch.save
            torch.save(obs, p)
            return str(p)
        except Exception as e:
            print(f"⚠️  failed to persist obs ep={episode_id} state={state_id}: {e}")
            return None

    def _persist_views_to_disk(self, episode_id: str, state_id: int, views_b64: dict[str, str]) -> dict[str, str]:
        """Persist base64 (data URL) JPEGs for each camera to disk.

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
                b64 = data_url[idx + len("base64,") :]
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
        """Load per-camera JPEG files and return data URLs."""
        if not view_paths:
            return {}
        out: dict[str, str] = {}
        for cam, path in view_paths.items():
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                out[cam] = f"data:image/jpeg;base64,{b64}"
            except Exception:
                # Missing/removed file -> skip this camera
                pass
        return out

    # =========================
    # Prompting and Demo Media Management
    # =========================

    def _prompts_root_dir(self) -> Path:
        """Root folder containing prompts/."""
        return (Path(__file__).resolve().parent / ".." / "data" / "prompts").resolve()

    def _task_dir(self, task_name: str | None = None) -> Path:
        tn = task_name or self.task_name()
        return (self._prompts_root_dir() / tn).resolve()

    def _parse_description_bank_entries(self, file_path: str) -> list[dict]:
        """Read description bank from file.

        Each line is a text prompt.
        Line number corresponds to video number.
        Returns: [{"id": int, "text": "<line content>", "full": "<line content>"}]

        """
        entries = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line_content = line.strip()
                    if line_content:  # Skip empty lines
                        entries.append({"id": line_num, "text": line_content, "full": line_content})
        except FileNotFoundError:
            print(f"Description bank file not found: {file_path}")
        except Exception as e:
            print(f"Error reading description bank file {file_path}: {e}")

        return entries

    def get_description_bank(self) -> dict:
        """Return both the raw description-bank text and its parsed entries.

        Reads from prompts/{task-name}/descriptions.txt where each line is a text prompt. Line number corresponds to
        video number.

        """
        task_name = self.task_name
        if not task_name:
            print("Warning: No task name set, cannot load description bank")
            return {"raw_text": "", "entries": []}

        # Construct file path: prompts/{task-name}/descriptions.txt
        file_path = self._task_dir(task_name) / "descriptions.txt"

        # Read raw text for compatibility
        raw_text = ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        except FileNotFoundError:
            print(f"Description bank file not found: {file_path}")
        except Exception as e:
            print(f"Error reading description bank file {file_path}: {e}")

        return {"raw_text": raw_text, "entries": self._parse_description_bank_entries(str(file_path))}

    # =========================
    # Simulation & Animation Management (Delegated to SimManager)
    # =========================

    def start_animation(
        self,
        session_id: str,
        goal_pose: dict = None,
        goal_joints: list = None,
        duration: float = 3.0,
        gripper_action: str = None,
    ) -> dict:
        """Start animation for a user session.

        Delegates to SimManager.

        """
        return self.sim_manager.start_animation(session_id, goal_pose, goal_joints, duration, gripper_action)

    def stop_animation(self, session_id: str) -> dict:
        """Stop animation for a user session.

        Delegates to SimManager.

        """
        return self.sim_manager.stop_animation(session_id)

    def get_animation_status(self) -> dict:
        """Get current animation status and availability.

        Delegates to SimManager.

        """
        return self.sim_manager.get_animation_status()

    def capture_animation_frame(self, session_id: str) -> dict:
        """Capture current animation frame for a user session.

        Delegates to SimManager.

        """
        return self.sim_manager.capture_animation_frame(session_id)

    def release_animation_session(self, session_id: str) -> dict:
        """Release animation slot for a disconnected session.

        Delegates to SimManager.

        """
        return self.sim_manager.release_animation_session(session_id)

    # =========================
    # Utility Methods
    # =========================
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

    def set_events(self, events):
        """Set the events object for keyboard-like control functionality."""
        self.events = events

    def load_main_cam_from_obs(self, obs: dict) -> np.ndarray | None:
        """Extract 'observation.images.cam_main' as RGB uint8 HxWx3; returns None if missing."""
        if not isinstance(obs, dict):
            return None
        for k in ("observation.images.cam_main", "observation.images.main", "observation.cam_main"):
            if k in obs:
                return self.obs_stream._to_uint8_rgb(obs[k])
        return None
