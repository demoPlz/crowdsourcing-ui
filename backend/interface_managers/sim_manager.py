"""
SimManager - Manages Isaac Sim integration for the crowd interface.

Handles:
- Persistent Isaac Sim worker lifecycle
- Sim capture queue and background worker
- Initial view capture from simulation
- Animation system (start, stop, status, frame capture, session management)
"""

import base64
import os
import queue
from pathlib import Path
from threading import Lock, Thread

JOINT_NAMES = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "left_carriage_joint"]


class SimManager:
    """Manages Isaac Sim worker and simulation capture."""

    def __init__(
        self,
        use_sim: bool,
        task_name: str | None,
        obs_cache_root: Path,
        state_lock: Lock,
        pending_states_by_episode: dict,
    ):
        """Initialize SimManager.

        Args:
            use_sim: Whether to use simulation
            task_name: Task name for USD file path
            obs_cache_root: Root directory for observation cache
            state_lock: Lock for thread-safe state updates
            pending_states_by_episode: Reference to pending states dict for atomic updates

        """
        self.use_sim = use_sim
        self.task_name = task_name
        self.obs_cache_root = obs_cache_root
        self.state_lock = state_lock
        self.pending_states_by_episode = pending_states_by_episode

        # Sim capture queue and worker thread
        self.sim_capture_queue = queue.Queue()
        self.sim_worker_thread = None
        self.sim_worker_running = False

        # Persistent Isaac Sim worker for reusable simulation
        self.isaac_manager = None

        # Start background worker
        self._start_sim_worker()

        # Start persistent Isaac Sim worker if using sim
        if self.use_sim:
            self._start_persistent_isaac_worker()

    def _start_sim_worker(self):
        """Start background thread for processing sim capture jobs."""
        self.sim_worker_thread = Thread(target=self._sim_worker, daemon=True)
        self.sim_worker_thread.start()

    def _start_persistent_isaac_worker(self):
        """Start persistent Isaac Sim worker using the new manager."""
        try:
            isaac_sim_path = os.environ.get("ISAAC_SIM_PATH")
            if not isaac_sim_path:
                print("⚠️ ISAAC_SIM_PATH not set, persistent worker disabled")
                return

            # Import the manager
            from isaac_sim.isaac_sim_worker_manager import PersistentWorkerManager

            self.isaac_manager = PersistentWorkerManager(
                isaac_sim_path=isaac_sim_path,
                output_base_dir=str(self.obs_cache_root / "persistent_isaac"),
                max_animation_users=1,  # Pre-clone 2 animation environments for development
            )

            initial_config = {
                "usd_path": f"public/assets/usd/{self.task_name}_flattened.usd",
                "robot_joints": [0.0] * 7,
                "object_poses": {},  # Will be populated from pose estimation
                "drawer_joint_positions": {},  # Will be populated from drawer tracking
            }

            print("🎥 Starting persistent Isaac Sim worker (this may take ~2 minutes)...")
            self.isaac_manager.start_worker(initial_config)
            print("✓ Persistent Isaac Sim worker ready")

            print("🎮 Initializing simulation and animation...")
            self.isaac_manager.capture_initial_state(initial_config)
            print("✓ Simulation and animation initialized")

        except Exception as e:
            print(f"⚠️ Failed to start persistent Isaac worker: {e}")
            self.isaac_manager = None

    def _sim_worker(self):
        """Background worker that processes sim capture jobs from the queue."""
        for work_item in iter(self.sim_capture_queue.get, None):
            if work_item is None:
                break

            episode_id = work_item["episode_id"]
            state_id = work_item["state_id"]
            state_info = work_item["state_info"]

            # Do the expensive sim capture
            sim_success = self.get_initial_views_from_sim(state_info)

            # Update the state atomically
            with self.state_lock:
                # Verify state still exists and update it
                if (
                    episode_id in self.pending_states_by_episode
                    and state_id in self.pending_states_by_episode[episode_id]
                ):
                    self.pending_states_by_episode[episode_id][state_id]["sim_ready"] = sim_success
                    print(
                        f"🎥 Sim capture {'completed' if sim_success else 'failed'} for episode {episode_id}, state {state_id}"
                    )

    def enqueue_sim_capture(self, episode_id: str, state_id: int, state_info: dict):
        """Enqueue a sim capture job for background processing.

        Args:
            episode_id: Episode identifier
            state_id: State identifier
            state_info: State info dict (will be updated when capture completes)

        """
        try:
            self.sim_capture_queue.put_nowait(
                {"episode_id": episode_id, "state_id": state_id, "state_info": state_info}
            )
            print(f"🎥 Queued sim capture for ep={episode_id}, state={state_id}")
        except Exception as e:
            print(f"⚠️  Failed to queue sim capture for ep={episode_id}, state={state_id}: {e}")

    def get_initial_views_from_sim(self, state_info) -> bool:
        """Use persistent worker for fast sim capture."""
        if not self.isaac_manager:
            print("⚠️ Isaac manager not available")
            return False

        try:
            # Extract state info
            joint_positions = state_info.get("joint_positions", {})
            episode_id = state_info.get("episode_id", "unknown")
            state_id = state_info.get("state_id", 0)
            left_carriage_external_force = state_info.get("left_carriage_external_force", 0.0)

            # Convert joint positions dict to list
            joint_positions_list = []
            for joint_name in JOINT_NAMES:
                joint_positions_list.append(joint_positions.get(joint_name, 0.0))

            # Get drawer joint positions if available
            drawer_joint_positions = state_info.get("drawer_joint_positions", {})
            
            print(f"🗄️  [SimManager] drawer_joint_positions from state_info: {drawer_joint_positions}")

            config = {
                "usd_path": f"public/assets/usd/{self.task_name}_flattened.usd",
                "robot_joints": joint_positions_list,
                "left_carriage_external_force": left_carriage_external_force,
                "object_poses": state_info.get("object_poses", {}),
                "drawer_joint_positions": drawer_joint_positions if drawer_joint_positions else {},
            }
            
            print(f"🗄️  [SimManager] Config drawer_joint_positions: {config['drawer_joint_positions']}")

            # Use persistent worker for fast capture with animation sync
            result = self.isaac_manager.update_state_and_sync_animations(config, f"ep_{episode_id}_state_{state_id}")

            if result.get("status") == "success":
                # Map Isaac Sim camera names to our expected names
                sim_result = result.get("result", {})
                sim_to_our_mapping = {"front_rgb": "front", "left_rgb": "left", "right_rgb": "right", "top_rgb": "top"}

                view_paths = {}
                for sim_name, our_name in sim_to_our_mapping.items():
                    if sim_name in sim_result:
                        view_paths[our_name] = sim_result[sim_name]

                if view_paths:
                    state_info["view_paths"] = view_paths
                    state_info["sim_ready"] = True
                    return True

            print(f"⚠️ Isaac capture failed: {result}")
            return False

        except Exception as e:
            print(f"⚠️ Isaac Sim capture failed: {e}")
            return False

    # =========================
    # Animation Management
    # =========================

    def start_animation(
        self,
        session_id: str,
        goal_pose: dict = None,
        goal_joints: list = None,
        duration: float = 3.0,
        gripper_action: str = None,
    ) -> dict:
        """Start animation for a user session."""
        if not self.use_sim or not self.isaac_manager:
            return {"status": "error", "message": "Simulation not available"}

        try:
            result = self.isaac_manager.start_user_animation_managed(
                session_id=session_id,
                goal_pose=goal_pose,
                goal_joints=goal_joints,
                duration=duration,
                gripper_action=gripper_action,
            )
            return result

        except Exception as e:
            print(f"⚠️ Animation start failed: {e}")
            return {"status": "error", "message": str(e)}

    def stop_animation(self, session_id: str) -> dict:
        """Stop animation for a user session."""
        if not self.use_sim or not self.isaac_manager:
            return {"status": "error", "message": "Simulation not available"}

        try:
            result = self.isaac_manager.stop_user_animation_managed(session_id)
            return result

        except Exception as e:
            print(f"⚠️ Animation stop failed: {e}")
            return {"status": "error", "message": str(e)}

    def get_animation_status(self) -> dict:
        """Get current animation status and availability."""
        if not self.use_sim or not self.isaac_manager:
            return {
                "available": False,
                "message": "Simulation not available",
                "animation_initialized": False,
                "max_users": 0,
                "available_slots": 0,
                "active_users": 0,
                "users": {},
            }

        try:
            status = self.isaac_manager.get_animation_status()
            status["available"] = True
            return status

        except Exception as e:
            print(f"⚠️ Animation status check failed: {e}")
            return {
                "available": False,
                "message": str(e),
                "animation_initialized": False,
                "max_users": 0,
                "available_slots": 0,
                "active_users": 0,
                "users": {},
            }

    def capture_animation_frame(self, session_id: str) -> dict:
        """Capture current animation frame for a user session."""
        if not self.use_sim or not self.isaac_manager:
            return {"status": "error", "message": "Simulation not available"}

        try:
            user_id = self.isaac_manager.get_user_by_session(session_id)
            if user_id is None:
                return {"status": "error", "message": "No active animation for session"}

            result = self.isaac_manager.capture_user_frame(user_id)

            if result.get("status") == "success" and "result" in result:
                # Convert file paths to base64 data URLs for frontend consumption
                frame_data = result["result"]
                base64_frames = {}

                for frame_key, file_path in frame_data.items():
                    try:
                        if os.path.exists(file_path):
                            with open(file_path, "rb") as f:
                                image_data = f.read()
                            b64 = base64.b64encode(image_data).decode("ascii")
                            base64_frames[frame_key] = f"data:image/jpeg;base64,{b64}"
                        else:
                            print(f"⚠️ Animation frame file not found: {file_path}")
                    except Exception as e:
                        print(f"⚠️ Error converting frame {frame_key} to base64: {e}")

                if base64_frames:
                    return {"status": "success", "result": base64_frames}
                else:
                    return {"status": "error", "message": "No frames could be loaded"}

            return result

        except Exception as e:
            print(f"⚠️ Animation frame capture failed: {e}")
            return {"status": "error", "message": str(e)}

    def release_animation_session(self, session_id: str) -> dict:
        """Release animation slot for a disconnected session."""
        if not self.use_sim or not self.isaac_manager:
            return {"status": "error", "message": "Simulation not available"}

        try:
            user_id = self.isaac_manager.get_user_by_session(session_id)
            if user_id is not None:
                success = self.isaac_manager.release_animation_slot(user_id)
                return {"status": "success" if success else "error", "released": success}
            else:
                return {"status": "success", "message": "No slot to release"}

        except Exception as e:
            print(f"⚠️ Animation session release failed: {e}")
            return {"status": "error", "message": str(e)}
