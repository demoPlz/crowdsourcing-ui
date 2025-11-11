"""State Manager Module.

Manages episode-based state lifecycle for the crowd interface. Handles state creation, labeling, auto-labeling, and
episode finalization.

"""

import queue
import random
from threading import Lock, Thread, Timer

import torch

# Joint names constant (shared with crowd_interface)
JOINT_NAMES = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "left_carriage_joint"]


class StateManager:
    """Manages episode-based state lifecycle for crowd interface.

    Responsibilities:
    - State creation and management (add_state, set_last_state_to_critical)
    - Response recording and completion tracking
    - Auto-labeling of non-critical states
    - Episode finalization and dataset saving
    - State monitoring and info retrieval

    Attributes:
        pending_states_by_episode: episode_id -> {state_id -> state_info}
        completed_states_by_episode: episode_id -> {state_id -> state_info}
        completed_states_buffer_by_episode: episode_id -> {state_id -> state_info} (for chronological dataset writes)
        current_serving_episode: Episode ID currently being served
        episodes_completed: Set of fully completed episode IDs
        next_state_id: Auto-incrementing state ID counter
        state_lock: Lock protecting all state data structures

    """

    def __init__(
        self,
        required_responses_per_state: int,
        required_responses_per_critical_state: int,
        autofill_critical_states: bool,
        num_autofill_actions: int,
        use_manual_prompt: bool,
        use_sim: bool,
        task_text: str | None,
        obs_cache_root,
        state_lock: Lock,
        pending_states_by_episode: dict,
        completed_states_by_episode: dict,
        completed_states_buffer_by_episode: dict,
        episode_finalize_grace_s: float,
        episodes_pending_save: set,
        # Managers for external dependencies
        obs_stream_manager,
        pose_estimation_manager,
        drawer_position_manager,
        sim_manager,
        # Callbacks for external operations
        persist_views_callback,
        persist_obs_callback,
        snapshot_views_callback,
        save_episode_callback,
    ):
        """Initialize state manager.

        Args:
            required_responses_per_state: Number of responses needed for non-critical states
            required_responses_per_critical_state: Number of responses needed for critical states
            autofill_critical_states: Whether to autofill critical states
            num_autofill_actions: Number of actions to autofill - 1
            use_manual_prompt: Whether manual prompting is enabled
            use_sim: Whether sim is enabled
            task_text: Task text for state info
            obs_cache_root: Root directory for observation cache
            state_lock: Lock protecting shared state data structures
            pending_states_by_episode: Shared dict of pending states
            completed_states_by_episode: Shared dict of completed states
            completed_states_buffer_by_episode: Shared dict for chronological dataset writes
            episode_finalize_grace_s: Grace period before finalizing empty episodes
            episodes_pending_save: Shared set of episodes pending save
            obs_stream_manager: ObservationStreamManager instance
            pose_estimation_manager: PoseEstimationManager instance
            drawer_position_manager: DrawerPositionManager instance
            sim_manager: SimManager instance
            persist_views_callback: Callback to persist views to disk
            persist_obs_callback: Callback to persist observations to disk
            snapshot_views_callback: Callback to snapshot current views
            save_episode_callback: Callback to save episode to dataset

        """
        self.required_responses_per_state = required_responses_per_state
        self.required_responses_per_critical_state = required_responses_per_critical_state
        self.autofill_critical_states = autofill_critical_states
        self.num_autofill_actions = num_autofill_actions
        self.use_manual_prompt = use_manual_prompt
        self.use_sim = use_sim
        self.task_text = task_text
        self._obs_cache_root = obs_cache_root

        # Shared state data structures (references)
        self.state_lock = state_lock
        self.pending_states_by_episode = pending_states_by_episode
        self.completed_states_by_episode = completed_states_by_episode
        self.completed_states_buffer_by_episode = completed_states_buffer_by_episode
        self._episodes_pending_save = episodes_pending_save

        # Episode state
        self.current_serving_episode = None
        self.episodes_completed = set()
        self.next_state_id = 0

        # Episode finalization
        self.episode_finalize_grace_s = episode_finalize_grace_s
        self._episode_finalize_timers: dict[str, Timer] = {}

        # Auto-labeling
        self.auto_label_queue = queue.Queue()
        self.auto_label_worker_thread = None

        # Manager dependencies
        self.obs_stream = obs_stream_manager
        self.pose_estimator = pose_estimation_manager
        self.drawer_position = drawer_position_manager
        self.sim_manager = sim_manager

        # Callbacks for external operations
        self._persist_views_callback = persist_views_callback
        self._persist_obs_callback = persist_obs_callback
        self._snapshot_views_callback = snapshot_views_callback
        self._save_episode_callback = save_episode_callback

        # Goal management
        self.latest_goal = None

        # Active episode tracking
        self._active_episode_id = None

        # Start auto-labeling worker
        self._start_auto_label_worker()

    # =========================
    # State Management (Public API - called externally)
    # =========================

    def add_state(
        self,
        joint_positions: dict,
        gripper_motion: int = None,
        obs_dict: dict[str, torch.Tensor] = None,
        episode_id: str = None,
        left_carriage_external_force: float | None = None,
    ):
        """Called by lerobot code to add states to backend."""
        joint_positions_float = {k: float(v) for k, v in joint_positions.items()}

        state_id = self.next_state_id
        self.next_state_id += 1

        # Persist views to disk to avoid storing in memory
        view_paths = self._persist_views_callback(episode_id, state_id, self._snapshot_views_callback())  # legacy

        obs_dict_deep_copy = {}
        for key, value in obs_dict.items():
            obs_dict_deep_copy[key] = value.clone().detach()
        obs_path = self._persist_obs_callback(episode_id, state_id, obs_dict_deep_copy)
        del obs_dict_deep_copy

        # Push obs to monitoring frontend
        self.obs_stream.push_obs_view("obs_main", obs_dict.get("observation.images.cam_main"))
        self.obs_stream.push_obs_view("obs_wrist", obs_dict.get("observation.images.cam_wrist"))

        state_info = {
            # Identity
            "state_id": state_id,
            "episode_id": episode_id,
            # Robot state
            "joint_positions": joint_positions_float,
            "gripper": gripper_motion,
            "controls": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],  # legacy, will remove
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
            "prompt_ready": False if self.use_manual_prompt else True,
            "text_prompt": None,  # replaces flex_text_prompt
            "video_prompt": None,  # replaces flex_video_id
            # Task
            "task_text": self.task_text,
            # Sim
            "sim_ready": False if self.use_sim else True,
            # Drawer joint positions will be computed when we call set_last_state_to_critical (like object_poses)
            # "drawer_joint_positions"
            # Poses of each object in self.object will be computed when we call set_last_state_to_critical
            # "object_poses"
            # No other fields; segmentation, and all others, no longer supported
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
        # ---- Phase 1: figure out which state to mark, under lock ----
        with self.state_lock:
            if not self.pending_states_by_episode:
                return

            latest_episode_id = max(self.pending_states_by_episode.keys())
            episode_states = self.pending_states_by_episode[latest_episode_id]
            if not episode_states:
                return

            latest_state_id = max(episode_states.keys())
            info = episode_states[latest_state_id]

            if info["critical"]:
                # Already set
                return

            info["critical"] = True
            self.demote_earlier_unanswered_criticals(latest_state_id, latest_episode_id)
            self.auto_label_previous_states(latest_state_id)

        # ---- Phase 2: enqueue pose jobs and BLOCK until all are reported ----
        poses_ready = self.pose_estimator.enqueue_pose_jobs_for_state(
            latest_episode_id, latest_state_id, info, wait=True, timeout_s=None
        )

        # ---- Phase 2.5: Estimate drawer position if tracking enabled ----
        drawer_positions_ready = False
        if self.drawer_position and self.drawer_position.enabled:
            try:
                # Load the observation
                obs_dict = torch.load(info["obs_path"], map_location="cpu")
                drawer_joint_positions = self.drawer_position.get_joint_position_from_obs(obs_dict)
                
                if drawer_joint_positions:
                    # Update the state info with drawer position
                    with self.state_lock:
                        ep = self.pending_states_by_episode.get(latest_episode_id)
                        if ep and latest_state_id in ep:
                            ep[latest_state_id]["drawer_joint_positions"] = drawer_joint_positions
                            drawer_positions_ready = True
                            print(f"🗄️  Drawer position captured for critical state (ep={latest_episode_id}, state={latest_state_id})")
                else:
                    print(f"⚠️  Drawer position estimation failed for ep={latest_episode_id}, state={latest_state_id}")
            except Exception as e:
                print(f"⚠️  Error estimating drawer position: {e}")

        # ---- Phase 3: only then consider sim ----
        with self.state_lock:
            # Re-lookup the state in case the dict changed
            ep = self.pending_states_by_episode.get(latest_episode_id)
            if not ep or latest_state_id not in ep:
                return
            info = ep[latest_state_id]

            if self.use_sim and poses_ready:
                info["sim_ready"] = False  # Mark as not ready initially
                self.sim_manager.enqueue_sim_capture(latest_episode_id, latest_state_id, info)
            else:
                # Not using sim, or poses not ready within timeout
                info["sim_ready"] = not self.use_sim
                if self.use_sim and not poses_ready:
                    print(
                        f"⏭️  Skipping/deferring sim capture: poses not ready for ep={latest_episode_id}, state={latest_state_id}"
                    )

    def get_latest_state(self) -> dict:
        """Get a pending state from current serving episode We only implement crowd mode, meaning that we serve the last
        state of the last episode always."""

        with self.state_lock:
            episode_id = self.current_serving_episode
            state_id = self.next_state_id - 1

            if (
                episode_id not in self.pending_states_by_episode
                or state_id not in self.pending_states_by_episode[episode_id]
                or not self.pending_states_by_episode[episode_id][state_id]["critical"]
            ):
                # No pending critical states left
                return {"status": "no_pending_states", "blocked_critical_states": False}

            state_info = self.pending_states_by_episode[episode_id][state_id]

            if state_info["critical"] and (not state_info["prompt_ready"] or not state_info["sim_ready"]):
                # There are pending states but no ready states

                return {
                    "status": "no_ready_states",
                    "blocked_critical_states": True,
                }

            # Return the latest state for labeling
            return state_info.copy()

    def record_response(self, response_data: dict):
        """Record a response for a specific state.

        Handles all the side-effects.

        """
        with self.state_lock:
            state_id = response_data["state_id"]
            episode_id = response_data["episode_id"]

            if (
                episode_id not in self.pending_states_by_episode
                or state_id not in self.pending_states_by_episode[episode_id]
            ):
                # State already fully labeled
                return

            state_info = self.pending_states_by_episode[episode_id][state_id]

            required_responses = (
                self.required_responses_per_critical_state
                if state_info["critical"]
                else self.required_responses_per_state
            )

            joint_positions = response_data["joint_positions"]
            gripper_action = response_data["gripper"]

            state_info["responses_received"] += 1

            goal_positions = []
            for joint_name in JOINT_NAMES:
                joint_value = joint_positions[joint_name]
                goal_positions.append(float(joint_value[0]))

            goal_positions[-1] = 0.044 if gripper_action > 0 else 0.0
            goal_positions = torch.tensor(goal_positions, dtype=torch.float32)
            state_info["actions"].append(goal_positions)

            # Autofill
            if state_info["critical"] and self.autofill_critical_states:
                remaining = state_info["responses_received"]
                clones_to_add = min(self.num_autofill_actions - 1, remaining)
                for _ in range(clones_to_add):
                    state_info["actions"].append(goal_positions.clone())
                state_info["responses_received"] += clones_to_add

            # Handle completion
            if state_info["responses_received"] >= required_responses:
                if state_info["critical"] and state_id == self.next_state_id - 1:
                    # Choose action to execute (a_execute) at random
                    # Shift chosen action to the front of the array
                    a_execute_index = random.randint(0, required_responses - 1)
                    state_info["actions"][0], state_info["actions"][a_execute_index] = (
                        state_info["actions"][a_execute_index],
                        state_info["actions"][0],
                    )
                    self.latest_goal = state_info["actions"][:required_responses][0]

                all_actions = torch.cat(state_info["actions"][:required_responses], dim=0)

                if required_responses < self.required_responses_per_critical_state:
                    # Pad unimportant states's action tensor
                    missing_responses = self.required_responses_per_critical_state - required_responses
                    action_dim = len(JOINT_NAMES)
                    padding_size = missing_responses * action_dim
                    padding = torch.full((padding_size,), float("nan"), dtype=torch.float32)
                    all_actions = torch.cat([all_actions, padding], dim=0)

                state_info["action_to_save"] = all_actions

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
        """Get episode-based state information for monitoring."""
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
                            if info.get("critical", False)
                            else self.required_responses_per_state
                        )
                        _txt = info.get("text_prompt")  # Updated field name
                        has_flex_text = bool(str(_txt or "").strip())
                        _vid = info.get("video_prompt")  # Updated field name
                        has_flex_video = _vid is not None

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
                        _txt = info.get("text_prompt")  # Updated field name
                        has_flex_text = bool(str(_txt or "").strip())
                        _vid = info.get("video_prompt")  # Updated field name
                        has_flex_video = _vid is not None

                        episode_states[state_id] = {
                            "responses_received": info["responses_received"],
                            "responses_needed": 0,  # Completed
                            "critical": bool(info.get("critical", False)),
                            "has_flex_text": has_flex_text,
                            "has_flex_video": has_flex_video,
                            "has_vlm_text": has_flex_text,  # legacy
                            "has_video_id": has_flex_video,  # legacy
                        }

                episodes_info[episode_id] = {
                    "states": episode_states,
                    "pending_count": len(self.pending_states_by_episode.get(episode_id, {})),
                    "completed_count": len(self.completed_states_by_episode.get(episode_id, {})),
                    "is_current_serving": episode_id == self.current_serving_episode,
                    "is_completed": episode_id in self.episodes_completed,
                    "pending_save": episode_id in self._episodes_pending_save,
                }

            return {
                "total_pending": total_pending,
                "current_serving_episode": self.current_serving_episode,
                "required_responses_per_state": self.required_responses_per_state,
                "required_responses_per_critical_state": self.required_responses_per_critical_state,
                "episodes": episodes_info,
            }

    def get_latest_goal(self) -> dict | None:
        """Get and clear the latest goal (for robot loop to consume)"""
        goal = self.latest_goal
        self.latest_goal = None
        return goal

    def set_active_episode(self, episode_id):
        """Mark which episode the outer robot loop is currently in (or None)."""
        with self.state_lock:
            self._active_episode_id = episode_id

    def set_prompt_ready(
        self, state_info: dict, episode_id: int, state_id: int, text: str | None, video_id: int | None
    ) -> None:
        """Set text/video prompt fields and mark as ready."""
        state_info["text_prompt"] = text  # Updated field name
        state_info["video_prompt"] = video_id  # Updated field name
        state_info["prompt_ready"] = True

        # Check if this is a critical state with "end." text - auto-fill with current position
        if text and text.strip().lower() == "end.":
            self._auto_fill_end_state_locked(state_info, episode_id, state_id)

    # =========================
    # Internal Helper Methods
    # =========================

    def demote_earlier_unanswered_criticals(self, current_state_id, episode_id):
        """Demote critical states before state_id in episode with episode_id to non-critical."""
        for state_id in self.pending_states_by_episode[episode_id].keys():
            if (
                state_id < current_state_id
                and self.pending_states_by_episode[episode_id][state_id]["critical"]
                and not self.pending_states_by_episode[episode_id][state_id]["actions"]
            ):
                self.pending_states_by_episode[episode_id][state_id]["critical"] = False

    def auto_label_previous_states(self, critical_state_id):
        self.auto_label_queue.put_nowait(critical_state_id)

    def _start_auto_label_worker(self):
        self.auto_label_worker_thread = Thread(target=self._auto_label_worker, daemon=True)
        self.auto_label_worker_thread.start()

    def _auto_label_worker(self):
        for critical_state_id in iter(self.auto_label_queue.get, None):
            self._auto_label(critical_state_id)

    def _auto_label(self, critical_state_id):
        """
        Given critical_state_id, auto-labels noncritical states in the same episode before the critical state with:"
        1. The executed action of the previous important state
        2. If no previous important state exists, the joint positions of the first state in the episode
        """
        with self.state_lock:
            episode_id = max(self.pending_states_by_episode.keys())

            episode_states = {
                **self.pending_states_by_episode[episode_id],
                **self.completed_states_by_episode[episode_id],
            }

            template_action = None

            previous_critical_id_in_episode = []
            for state_id in episode_states.keys():
                if (
                    episode_states[state_id]["critical"]
                    and state_id < critical_state_id
                    and len(episode_states[state_id]["actions"]) > 0
                ):
                    previous_critical_id_in_episode.append(state_id)

            if previous_critical_id_in_episode:  # Previous critical states exist
                latest_critical_state = episode_states[max(previous_critical_id_in_episode)]
                template_action = latest_critical_state["actions"][0]
            else:  # This is the first critical state in the episode
                first_state_id = min(episode_states.keys())
                first_state = episode_states[first_state_id]
                # Direct access to joint_positions and gripper in flattened structure
                joint_positions = first_state["joint_positions"]
                gripper_action = first_state["gripper"]
                goal_positions = []
                for joint_name in JOINT_NAMES:
                    joint_value = joint_positions[joint_name]
                    goal_positions.append(float(joint_value))

                goal_positions[-1] = 0.044 if gripper_action > 0 else 0.0
                template_action = torch.tensor(goal_positions, dtype=torch.float32)

            states_to_label = []
            for state_id, state_info in episode_states.items():
                if (
                    state_id < critical_state_id
                    and not state_info["critical"]
                    and state_id not in self.completed_states_by_episode[episode_id]
                ):
                    states_to_label.append(state_id)

            for state_id in states_to_label:
                state_info = episode_states[state_id]

                while state_info["responses_received"] < self.required_responses_per_state:
                    state_info["actions"].append(template_action.clone())
                    state_info["responses_received"] += 1

                all_actions = torch.cat(state_info["actions"][: self.required_responses_per_state], dim=0)

                # Pad with inf values to match critical state shape
                missing_responses = self.required_responses_per_critical_state - self.required_responses_per_state
                action_dim = len(JOINT_NAMES)
                padding_size = missing_responses * action_dim
                padding = torch.full((padding_size,), float("nan"), dtype=torch.float32)
                all_actions = torch.cat([all_actions, padding], dim=0)

                state_info["action_to_save"] = all_actions

                # Save to completed_states buffer (for forming training set)
                if episode_id not in self.completed_states_buffer_by_episode:
                    self.completed_states_buffer_by_episode[episode_id] = {}
                self.completed_states_buffer_by_episode[episode_id][state_id] = state_info

                # Save to completed states (for monitoring)
                if episode_id not in self.completed_states_by_episode:
                    self.completed_states_by_episode[episode_id] = {}
                self.completed_states_by_episode[episode_id][state_id] = state_info

                del self.pending_states_by_episode[episode_id][state_id]

    def _schedule_episode_finalize_after_grace(self, episode_id: int):
        delay = self.episode_finalize_grace_s
        timer = Timer(delay, self._finalize_episode_if_still_empty, args=(episode_id,))
        timer.daemon = True
        self._episode_finalize_timers[episode_id] = timer
        timer.start()

    def _finalize_episode_if_still_empty(self, episode_id: int):
        """Timer callback."""
        with self.state_lock:
            self._episode_finalize_timers.pop(episode_id, None)

            if self.pending_states_by_episode.get(episode_id):
                # New states has become pending in the episode
                return

            self.episodes_completed.add(episode_id)  # for monitoring

            buffer = self.completed_states_buffer_by_episode[episode_id]
            self._save_episode_callback(buffer)

            del self.completed_states_buffer_by_episode[episode_id]

    def _auto_fill_end_state_locked(self, state_info: dict, episode_id: int, state_id: int) -> None:
        """Auto-fill an critical state labeled as "end." with multiple copies of its current position.

        MUST be called with self.state_lock already held.

        """
        # Direct access to joint positions and gripper in flattened structure
        joint_positions = state_info.get("joint_positions", {})
        gripper_action = state_info.get("gripper", 0)

        # Convert joint positions to action tensor (same as autolabel logic)
        goal_positions = []
        for joint_name in JOINT_NAMES:
            v = joint_positions.get(joint_name, 0.0)
            v = float(v[0]) if isinstance(v, (list, tuple)) and len(v) > 0 else float(v)
            goal_positions.append(v)
        # Set gripper position based on gripper action
        goal_positions[-1] = 0.044 if gripper_action > 0 else 0.0

        position_action = torch.tensor(goal_positions, dtype=torch.float32)

        state_info["actions"] = [position_action for _ in range(self.required_responses_per_critical_state)]
        all_actions = torch.cat(state_info["actions"][: self.required_responses_per_critical_state], dim=0)

        state_info["action_to_save"] = all_actions

        self.completed_states_buffer_by_episode[episode_id][state_id] = state_info
        self.completed_states_by_episode[episode_id][state_id] = state_info

        del self.pending_states_by_episode[episode_id][state_id]

        if not self.pending_states_by_episode[episode_id]:
            self._schedule_episode_finalize_after_grace(episode_id)
