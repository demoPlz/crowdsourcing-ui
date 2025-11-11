#!/usr/bin/env python3
"""
Isaac Sim worker script with two modes:
1. Static image capture (for frontend interaction)
2. Animation mode (for physics simulation with direct joint control)
"""

import os
import time
import traceback

from PIL import Image


class AnimationFrameCache:
    """Cache system for storing and replaying animation frames efficiently."""

    def __init__(self, user_id: int, duration: float, fps: float = 30.0):
        self.user_id = user_id
        self.duration = duration
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.total_frames = int(duration * fps)

        # Frame storage: frame_index -> {camera_name: image_path}
        self.frames = {}
        self.frame_count = 0
        self.is_complete = False
        self.generation_start_time = None
        self.replay_start_time = None
        self.current_replay_frame = 0

        # Track which cameras we have
        self.camera_names = set()

    def start_generation(self):
        """Mark the start of frame generation."""
        self.generation_start_time = time.time()
        self.frames.clear()
        self.frame_count = 0
        self.is_complete = False
        print(f"📹 Starting frame generation for user {self.user_id} - {self.total_frames} frames at {self.fps} FPS")

    def add_frame(self, frame_index: int, camera_data: dict):
        """Add a frame to the cache
        Args:
            frame_index: The frame number (0-based)
            camera_data: Dict of {camera_name: image_path}
        """
        self.frames[frame_index] = camera_data.copy()
        self.camera_names.update(camera_data.keys())
        self.frame_count += 1

        if frame_index == self.total_frames - 1:
            self.is_complete = True
            if self.generation_start_time is not None:
                generation_time = time.time() - self.generation_start_time
                print(
                    f"✅ Frame generation complete for user {self.user_id}: {self.frame_count} frames in {generation_time:.2f}s"
                )
            else:
                print(
                    f"✅ Frame generation complete for user {self.user_id}: {self.frame_count} frames (no timing available)"
                )

    def get_current_replay_frame(self) -> dict | None:
        """Get the current frame for replay based on elapsed time."""
        if not self.is_complete:
            return None

        if self.replay_start_time is None:
            self.replay_start_time = time.time()
            self.current_replay_frame = 0

        # Calculate which frame we should be showing
        elapsed = time.time() - self.replay_start_time
        target_frame = int((elapsed % self.duration) * self.fps)

        # If we've looped, reset timing for smoother looping
        if target_frame < self.current_replay_frame:
            self.replay_start_time = time.time()
            target_frame = 0

        self.current_replay_frame = target_frame

        # Return the frame data
        return self.frames.get(target_frame)

    def reset_replay(self):
        """Reset replay to start from beginning."""
        self.replay_start_time = None
        self.current_replay_frame = 0

    def clear_cache(self):
        """Clear all cached frames and clean up files."""
        if self.frames:
            print(f"🧹 Clearing frame cache for user {self.user_id} - {len(self.frames)} frames")

            # Delete frame files from disk
            for frame_data in self.frames.values():
                for image_path in frame_data.values():
                    try:
                        if os.path.exists(image_path):
                            os.remove(image_path)
                    except Exception as e:
                        print(f"Warning: Could not delete frame file {image_path}: {e}")

        self.frames.clear()
        self.frame_count = 0
        self.is_complete = False
        self.camera_names.clear()
        self.generation_start_time = None
        self.replay_start_time = None
        self.current_replay_frame = 0


class IsaacSimWorker:
    GRIPPER_LEFT_IDX = 6
    GRIPPER_RIGHT_IDX = 7

    def __init__(self, simulation_app=None):
        self.world = None
        self.robot = None
        self.cameras = {}
        self.user_environments = {}
        self.active_animations = {}
        self.animation_mode = False
        self.running = True
        # State management for reuse
        self.simulation_initialized = False
        self.objects = {}  # Store object references for reuse
        self.hide_robot_funcs = None  # Store hide/show functions
        self.simulation_app = simulation_app  # Store simulation app reference
        self.last_sync_config = None  # Store last synchronized config for animation reset

        # === Frame cache system for efficient animation replay ===
        self.frame_caches = {}  # user_id -> AnimationFrameCache
        self.frame_generation_in_progress = set()  # Track which users are generating frames
        self.animation_stop_requested = set()  # Track users for whom stop has been requested during generation
        self.worker_communication_dir = None  # Will be set by persistent worker for direct command checking

        # === Chunked frame generation system ===
        self.chunked_generation_state = {}  # user_id -> generation state for async processing

    def initialize_simulation(self, config):
        """One-time simulation setup that can be reused across state updates."""
        if self.simulation_initialized:
            return

        import carb
        import numpy as np
        import omni.usd
        from isaacsim.core.utils.prims import get_prim_at_path, set_prim_visibility
        from isaacsim.sensors.camera import Camera, get_all_camera_objects
        from omni.isaac.core import World
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.prims import RigidPrim, XFormPrim
        from pxr import UsdPhysics

        # Configuration
        USD_PATH = config["usd_path"]
        ROBOT_PATH = "/World/wxai"
        OBJ_Cube_Blue_PATH = "/World/Cube_Blue"
        OBJ_Cube_Red_PATH = "/World/Cube_Red"
        OBJ_TENNIS_PATH = "/World/Tennis"

        # Load the USD stage (only once)
        print(f"Loading environment from {USD_PATH}")
        omni.usd.get_context().open_stage(USD_PATH)

        # Wait for the stage to load
        for i in range(20):
            if self.simulation_app:
                self.simulation_app.update()
            else:
                # Fallback to global if not provided (for backward compatibility)
                import sys

                simulation_app = getattr(sys.modules[__name__], "simulation_app", None)
                if simulation_app:
                    simulation_app.update()

        # Create the World object (only once)
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        print("Stage loaded and World object created.")

        # Get handles to the prims (store for reuse)
        self.robot = self.world.scene.add(Articulation(prim_path=ROBOT_PATH, name="widowx_robot"))
        self.robot_prim = get_prim_at_path(ROBOT_PATH)
        self.objects["Cube_Blue"] = self.world.scene.add(RigidPrim(prim_path=OBJ_Cube_Blue_PATH, name="Cube_Blue"))
        self.objects["Cube_Red"] = self.world.scene.add(RigidPrim(prim_path=OBJ_Cube_Red_PATH, name="Cube_Red"))
        self.objects["tennis_ball"] = self.world.scene.add(RigidPrim(prim_path=OBJ_TENNIS_PATH, name="tennis_ball"))

        # Store drawer reference for joint manipulation (don't load as Articulation if not properly set up)
        self.drawer_prim_path = "/World/drawer_shell"

        import omni.usd
        from omni.isaac.core.prims import XFormPrim

        stage = omni.usd.get_context().get_stage()  # just to confirm they exist; not modifying

        for path, key in [
            ("/World/tray_01", "tray_01"),
            ("/World/tray_02", "tray_02"),
            ("/World/tray_03", "tray_03"),
        ]:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                print(f"⚠️ Missing prim: {path} (skipping)")
                self.objects[key] = None
                continue

            # Important: XFormPrim wrapper only. We do NOT set any pose here.
            self.objects[key] = self.world.scene.add(XFormPrim(prim_path=path, name=key))
            print(f"✓ Registered as-authored: {path} → objects['{key}']")

        # Get cameras (only once)
        stage = omni.usd.get_context().get_stage()
        all_cameras = get_all_camera_objects(root_prim="/")
        self.cameras = {all_cameras[i].name: all_cameras[i] for i in range(len(all_cameras))}

        # Reset world and initialize cameras (only once)
        self.world.reset()

        # Initialize cameras (only once)
        for camera in all_cameras:
            camera.initialize()
            camera.set_resolution((640, 480))
            camera.add_rgb_to_frame()

        # stage.GetPrimAtPath("/World/wxai/joints/right_carriage_joint").GetAttribute('drive:linear:physics:stiffness').Set(50000.0)

        # Create robot hide/show functions (only once)
        def hide_robot():
            """Alternative: Move robot far away (preserves everything)"""
            set_prim_visibility(self.robot_prim, False)

        def show_robot():
            """Restore robot to original position."""
            set_prim_visibility(self.robot_prim, True)

        self.hide_robot_funcs = {"hide": hide_robot, "show": show_robot}

        self.simulation_initialized = True
        print("Simulation initialized successfully - ready for state updates")

    def set_robot_joints(self):
        import numpy as np

        # Detect grasp
        gripper_external_force = self.last_sync_config.get("left_carriage_external_force", 0)
        grasped = gripper_external_force > 50  # GRASPED_THRESHOLD
        robot_joints = self.last_sync_config["robot_joints"]
        robot_joints_open_gripper = robot_joints.copy()

        if grasped:
            robot_joints_open_gripper[-1] = 0.044  # left finger open
            # do not touch the mimic DOF here

        robot_joints_open_gripper_8dof = np.append(robot_joints_open_gripper, robot_joints_open_gripper[-1])

        self.robot.set_joint_positions(robot_joints_open_gripper_8dof)

        from omni.isaac.core.utils.types import ArticulationAction

        self.robot.set_joint_velocities(np.zeros(7, dtype=float), joint_indices=list(range(7)))

        # Let physics settle after initial positioning
        for step in range(20):
            self.world.step(render=True)

        # PHYSICS-BASED GRIPPER CLOSING: If grasp was detected, smoothly close gripper
        if grasped:
            # Create target position with original (closed) gripper values
            target_q = robot_joints.copy()  # This has the original closed gripper positions

            # Apply smooth closing action over several steps for stable grasp
            for close_step in range(15):  # ~0.5 seconds at 30Hz
                current_q = self.robot.get_joint_positions()
                alpha = (close_step + 1) / 15.0

                # Interpolate ONLY the left finger (index 6)
                new_left = current_q[6] + alpha * (target_q[6] - current_q[6])

                self.robot.apply_action(
                    ArticulationAction(
                        joint_positions=[new_left, new_left],
                        joint_indices=[self.GRIPPER_LEFT_IDX, self.GRIPPER_RIGHT_IDX],
                    )
                )

                # Step physics
                self.world.step(render=True)

        # Final physics settling
        for step in range(3):
            self.world.step(render=True)

    def set_drawer_joints(self):
        """Set drawer joint positions from config using direct USD API"""
        import omni.usd

        # Get drawer joint positions from config
        drawer_joint_positions = self.last_sync_config.get("drawer_joint_positions", {})
        
        if not drawer_joint_positions:
            # No drawer positions specified, keep at default (closed)
            print(f"[Worker] 🗄️  No drawer joint positions in config, keeping at closed position")
            return
        
        # Get the joint position for Drawer_Joint
        drawer_joint_pos = drawer_joint_positions.get("Drawer_Joint", 0.0)
        
        print(f"[Worker] 🗄️  Setting drawer joint: Drawer_Joint = {drawer_joint_pos:.4f} m ({abs(drawer_joint_pos)*100:.2f} cm {'open' if drawer_joint_pos < 0 else 'closed'})")
        
        if drawer_joint_pos == 0.0:
            # Already at closed position, no need to update
            print(f"[Worker] 🗄️  Drawer at closed position (0.0), skipping update")
            return
        
        # Set the drawer joint position using USD API
        stage = omni.usd.get_context().get_stage()
        joint_prim = stage.GetPrimAtPath("/World/Drawer_Joint")
        
        if not joint_prim or not joint_prim.IsValid():
            print(f"⚠️ Could not find Drawer_Joint at /World/Drawer_Joint")
            return
        
        # Set the joint position using the physics drive target
        # For prismatic joints, this is typically the drive:linear:physics:targetPosition attribute
        try:
            if joint_prim.HasAttribute("drive:linear:physics:targetPosition"):
                joint_prim.GetAttribute("drive:linear:physics:targetPosition").Set(drawer_joint_pos)
                print(f"[Worker] ✓ Set drawer joint target position to {drawer_joint_pos:.4f} via drive:linear:physics:targetPosition")
            elif joint_prim.HasAttribute("physics:position"):
                joint_prim.GetAttribute("physics:position").Set(drawer_joint_pos)
                print(f"[Worker] ✓ Set drawer joint position to {drawer_joint_pos:.4f} via physics:position")
            else:
                # List available attributes for debugging
                attrs = [attr.GetName() for attr in joint_prim.GetAttributes()]
                print(f"[Worker] ⚠️ Could not find position attribute on joint. Available attributes: {attrs[:10]}...")
                
        except Exception as e:
            print(f"[Worker] ⚠️ Failed to set drawer joint position: {e}")
        
        # Let physics settle after positioning
        for step in range(10):
            self.world.step(render=True)

    def update_state(self, config):
        """Update robot joints and object poses without recreating simulation."""
        import numpy as np

        if not self.simulation_initialized:
            raise RuntimeError("Must call initialize_simulation() first")

        self.robot.set_joint_positions(np.zeros(8, dtype=float))

        # Store the config
        self.last_sync_config = config.copy()
        self.last_sync_config["robot_joints"] = np.array(self.last_sync_config["robot_joints"])
        # Clone last dimension for mimic join
        object_states = config.get(
            "object_poses",
            {
                "Cube_Blue": {"pos": [0.6, 0.0, 0.1], "rot": [0, 0, 0, 1]},
                "Cube_Red": {"pos": [0.6, 0.2, 0.1], "rot": [0, 0, 0, 1]},
                "Tennis": {"pos": [0.6, -0.2, 0.1], "rot": [0, 0, 0, 1]},
            },
        )
        
        print(f"[Worker] 📦 Updating object poses:")
        for obj_name, pose in object_states.items():
            if pose:
                print(f"[Worker]    {obj_name}: pos={pose.get('pos', 'N/A')}")

        # Update object poses FIRST (before robot positioning)
        if self.objects["Cube_Blue"].is_valid():
            state = object_states.get("Cube_Blue")
            if state:
                pos = np.array(state["pos"])
                rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])
                self.objects["Cube_Blue"].set_world_pose(position=pos, orientation=rot)

        if self.objects["Cube_Red"].is_valid():
            state = object_states.get("Cube_Red")
            if state:
                pos = np.array(state["pos"])
                rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])
                self.objects["Cube_Red"].set_world_pose(position=pos, orientation=rot)

        if self.objects["tennis_ball"].is_valid():
            state = object_states.get("Tennis")
            if state:
                pos = np.array(state["pos"])
                rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])
                self.objects["tennis_ball"].set_world_pose(position=pos, orientation=rot)

        # Let physics settle after object positioning
        for step in range(20):
            self.world.step(render=True)

        # Set drawer joint positions (before robot, so drawer is in correct state)
        self.set_drawer_joints()

        self.set_robot_joints()

    def capture_current_state_images(self, output_dir):
        """Capture images with current state (robot temporarily hidden for capture only)"""
        import os

        from PIL import Image

        if not self.simulation_initialized:
            raise RuntimeError("Must call initialize_simulation() first")

        os.makedirs(output_dir, exist_ok=True)

        print("Capturing images with current state...")

        # Let physics settle with robot hidden
        for step in range(10):
            self.world.step(render=True)  # Need render=True before camera capture

        # Temporarily hide robot for static capture
        self.hide_robot_funcs["hide"]()

        # Let physics settle with robot hidden
        for step in range(10):
            self.world.step(render=True)  # Need render=True before camera capture

        # Capture static images
        front_rgb = self.cameras["Camera_Front"].get_rgb()
        left_rgb = self.cameras["Camera_Left"].get_rgb()
        right_rgb = self.cameras["Camera_Right"].get_rgb()
        top_rgb = self.cameras["Camera_Top"].get_rgb()

        # IMPORTANT: Restore robot visibility after capture - environment should always have robot visible
        self.hide_robot_funcs["show"]()

        # Let physics settle with robot restored
        for step in range(10):
            self.world.step(render=True)  # After capture, render=True is fine

        # Save static images
        Image.fromarray(front_rgb).save(f"{output_dir}/static_front_image.jpg", "JPEG", quality=90)
        Image.fromarray(left_rgb).save(f"{output_dir}/static_left_image.jpg", "JPEG", quality=90)
        Image.fromarray(right_rgb).save(f"{output_dir}/static_right_image.jpg", "JPEG", quality=90)
        Image.fromarray(top_rgb).save(f"{output_dir}/static_top_image.jpg", "JPEG", quality=90)

        return {
            "front_rgb": f"{output_dir}/static_front_image.jpg",
            "left_rgb": f"{output_dir}/static_left_image.jpg",
            "right_rgb": f"{output_dir}/static_right_image.jpg",
            "top_rgb": f"{output_dir}/static_top_image.jpg",
            "status": "static_images_captured",
        }

    def capture_static_images(self, config, output_dir):
        """Mode 1: Capture static images (robot hidden) for frontend
        Now uses reusable simulation initialization"""

        # Initialize simulation if not already done
        if not self.simulation_initialized:
            self.initialize_simulation(config)

        # Update to new state
        self.update_state(config)

        # Capture images with current state
        return self.capture_current_state_images(output_dir)

    def update_and_capture(self, config, output_dir):
        """Convenience method: Update state and capture images in one call
        Use this for state cycling without reinstantiation"""

        if not self.simulation_initialized:
            # First call - initialize everything
            return self.capture_static_images(config, output_dir)
        else:
            # Subsequent calls - just update state and capture
            self.update_state(config)
            return self.capture_current_state_images(output_dir)

    def initialize_animation_mode(self, max_users=8):
        """Mode 2: Initialize cloned environments for animation"""
        try:
            import numpy as np
            from omni.isaac.cloner import Cloner
            from omni.isaac.core.articulations import Articulation

            print(f"Initializing animation mode with {max_users} user environments...")

            # Check prerequisites
            if not self.simulation_initialized:
                raise RuntimeError("Simulation must be initialized before animation mode")
            if not self.world:
                raise RuntimeError("World object not available")
            if not self.robot:
                raise RuntimeError("Robot object not available")

            # Ensure robot is visible and positioned correctly
            if self.hide_robot_funcs:
                self.hide_robot_funcs["show"]()
                print("✅ Robot is visible and ready for animation mode")

            # Let physics settle
            for step in range(10):
                self.world.step(render=True)

            print("Animation mode will use direct joint control only")

            # Try minimal cloning approach - clone only the essentials
            cloner = Cloner()
            environment_spacing = 50.0  # Larger spacing to avoid any interactions

            for user_id in range(max_users):
                if user_id == 0:
                    # User 0 uses original environment
                    target_path = "/World"
                    user_robot = self.robot
                    user_cameras = self.cameras
                    print(f"User {user_id}: Using original environment")

                else:
                    # Try cloning with larger spacing and minimal approach
                    target_path = f"/Env_{user_id}"  # Shorter path
                    offset = [
                        user_id * environment_spacing,
                        user_id * environment_spacing,
                        0,
                    ]  # X and Y axis offset for diagonal spacing

                    print(f"User {user_id}: Attempting minimal clone at offset {offset}")

                    try:
                        # Clone with very simple approach
                        cloner.clone(
                            source_prim_path="/World",
                            prim_paths=[target_path],
                            positions=[np.array(offset)],
                            copy_from_source=True,
                        )

                        # Let physics settle
                        for step in range(20):
                            self.world.step(render=True)

                        # Get robot
                        robot_path = f"{target_path}/wxai"
                        user_robot = self.world.scene.add(
                            Articulation(prim_path=robot_path, name=f"robot_user_{user_id}")
                        )

                        # Register cloned objects in scene registry for easy access
                        from omni.isaac.core.prims import RigidPrim

                        try:
                            Cube_Blue_path = f"{target_path}/Cube_Blue"
                            Cube_Red_path = f"{target_path}/Cube_Red"
                            tennis_path = f"{target_path}/Tennis"

                            self.world.scene.add(RigidPrim(prim_path=Cube_Blue_path, name=f"Cube_Blue_user_{user_id}"))
                            self.world.scene.add(RigidPrim(prim_path=Cube_Red_path, name=f"Cube_Red_user_{user_id}"))
                            self.world.scene.add(RigidPrim(prim_path=tennis_path, name=f"tennis_user_{user_id}"))
                            print(f"✅ Registered cloned objects for user {user_id}")
                        except Exception as obj_e:
                            print(f"⚠️ Failed to register cloned objects for user {user_id}: {obj_e}")

                        # Get cameras
                        user_cameras = {}
                        from isaacsim.sensors.camera import get_all_camera_objects

                        all_user_cameras = get_all_camera_objects(root_prim=target_path)
                        for camera in all_user_cameras:
                            camera.initialize()
                            camera.set_resolution((640, 480))
                            camera.add_rgb_to_frame()
                            user_cameras[camera.name] = camera

                        print(f"✅ Successfully cloned environment for user {user_id}")

                    except Exception as e:
                        print(f"⚠️ Clone failed for user {user_id}: {e}")
                        print(f"Falling back to shared environment for user {user_id}")
                        # Fallback to shared environment
                        target_path = "/World"
                        user_robot = self.robot
                        user_cameras = self.cameras

                # Store environment data
                self.user_environments[user_id] = {
                    "robot": user_robot,
                    "cameras": user_cameras,
                    "world_path": target_path,
                    "spatial_offset": [0, 0, 0],
                }

            self.animation_mode = True
            print(f"Animation mode initialized with {max_users} environments")

        except Exception as e:
            print(f"❌ Animation mode initialization failed: {e}")
            import traceback

            traceback.print_exc()
            self.animation_mode = False
            self.user_environments = {}
            raise e  # Re-raise so the caller knows it failed

    def sync_animation_environments(self, config):
        """Synchronize all animation environments to match the current state."""
        if not self.animation_mode:
            print("Animation mode not initialized, skipping sync")
            return

        import numpy as np
        import omni.usd
        from pxr import Gf, UsdGeom

        print("Synchronizing animation environments to new state...")

        object_states = config.get(
            "object_poses",
            {
                "Cube_Blue": {"pos": [0.6, 0.0, 0.1], "rot": [0, 0, 0, 1]},
                "Cube_Red": {"pos": [0.6, 0.2, 0.1], "rot": [0, 0, 0, 1]},
                "Tennis": {"pos": [0.6, -0.2, 0.1], "rot": [0, 0, 0, 1]},
            },
        )

        # Update each animation environment
        for user_id, env_data in self.user_environments.items():
            try:
                self.robot.set_joint_positions(np.zeros(8, dtype=float))

                # Update object poses for ALL environments using appropriate object references
                if user_id == 0:
                    # User 0: Use original object references - reset to absolute positions
                    print(f"🔧 Syncing objects for user 0 (original environment)")

                    if "Cube_Blue" in object_states:
                        state = object_states["Cube_Blue"]
                        if state is not None:  # Skip if pose estimation failed
                            pos = np.array(state["pos"])
                            rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])
                            self.objects["Cube_Blue"].set_world_pose(position=pos, orientation=rot)
                            self.objects["Cube_Blue"].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                            self.objects["Cube_Blue"].set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                            print(f"✅ Synced Cube_Blue to {pos} (physics cleared)")

                    if "Cube_Red" in object_states:
                        state = object_states["Cube_Red"]
                        if state is not None:  # Skip if pose estimation failed
                            pos = np.array(state["pos"])
                            rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])
                            self.objects["Cube_Red"].set_world_pose(position=pos, orientation=rot)
                            self.objects["Cube_Red"].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                            self.objects["Cube_Red"].set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                            print(f"✅ Synced Cube_Red to {pos} (physics cleared)")

                    if "Tennis" in object_states:
                        state = object_states["Tennis"]
                        if state is not None:  # Skip if pose estimation failed
                            pos = np.array(state["pos"])
                            rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])
                            self.objects["tennis_ball"].set_world_pose(position=pos, orientation=rot)
                            self.objects["tennis_ball"].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                            self.objects["tennis_ball"].set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                            print(f"✅ Synced Tennis to {pos} (physics cleared)")

                else:
                    # Cloned environments: Sync objects using scene registry WITH SPATIAL OFFSET
                    print(
                        f"🔧 Syncing objects for user {user_id} (cloned environment) using scene registry WITH OFFSET"
                    )

                    # Calculate the spatial offset for this environment
                    # User environments are spaced diagonally: [user_id * 50, user_id * 50, 0]
                    environment_spacing = 50.0
                    spatial_offset = np.array([user_id * environment_spacing, user_id * environment_spacing, 0])
                    print(f"📍 User {user_id} spatial offset: {spatial_offset}")

                    # Sync each cloned object using scene registry with spatial offset applied
                    object_mappings = [
                        ("Cube_Blue", f"Cube_Blue_user_{user_id}"),
                        ("Cube_Red", f"Cube_Red_user_{user_id}"),
                        ("Tennis", f"tennis_user_{user_id}"),
                    ]

                    for config_key, scene_name in object_mappings:
                        if config_key in object_states:
                            state = object_states[config_key]
                            if state is None:  # Skip if pose estimation failed
                                continue

                            # CRITICAL: Apply spatial offset to object position
                            original_pos = np.array(state["pos"])
                            offset_pos = original_pos + spatial_offset
                            rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])

                            print(
                                f"🔄 {config_key}: Original pos {original_pos} + offset {spatial_offset} = {offset_pos}"
                            )

                            # Set object position using scene registry with offset
                            if self.world.scene.object_exists(scene_name):
                                scene_obj = self.world.scene.get_object(scene_name)
                                scene_obj.set_world_pose(position=offset_pos, orientation=rot)
                                scene_obj.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                                scene_obj.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                                print(
                                    f"✅ Synced {config_key} via scene registry to offset pos: {offset_pos} (physics cleared)"
                                )
                            else:
                                print(f"⚠️ Scene object {scene_name} not found for user {user_id} - skipping sync")

                    print(f"📍 User {user_id} objects synced using scene registry WITH SPATIAL OFFSET")

                for step in range(50):
                    self.world.step(render=True)

                self.set_robot_joints()

            except Exception as e:
                print(f"Warning: Failed to sync environment {user_id}: {e}")

        # Let physics settle across all environments
        # for step in range(10):
        #     self.world.step(render=True)

        print("Animation environment synchronization complete")

    def _update_object_transform(self, stage, object_path, position, rotation):
        """Helper method to update object transform with proper USD precision handling."""
        from pxr import Gf, UsdGeom

        prim = stage.GetPrimAtPath(object_path)
        if not prim.IsValid():
            return

        xformable = UsdGeom.Xformable(prim)

        # Get or create transform operations with matching precision
        xform_ops = xformable.GetOrderedXformOps()
        translate_op = None
        orient_op = None

        # Find existing ops
        for op in xform_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                orient_op = op

        # Create ops if they don't exist
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        if orient_op is None:
            orient_op = xformable.AddOrientOp()

        # Set translation and rotation with proper precision matching
        translate_op.Set((position[0], position[1], position[2]))

        # Detect the precision type of the existing orient operation and match it
        if orient_op is not None:
            # Get the attribute to check its type
            attr = orient_op.GetAttr()
            type_name = attr.GetTypeName()

            # Use the appropriate quaternion type based on existing precision
            if "quatd" in str(type_name):
                # Double precision quaternion
                quat = Gf.Quatd(rotation[3], rotation[0], rotation[1], rotation[2])  # Gf.Quatd(w, x, y, z)
            else:
                # Float precision quaternion (default)
                quat = Gf.Quatf(rotation[3], rotation[0], rotation[1], rotation[2])  # Gf.Quatf(w, x, y, z)

            orient_op.Set(quat)

    def _apply_environment_offset(self, environment_path, offset):
        """Apply spatial offset to an entire cloned environment."""
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        env_prim = stage.GetPrimAtPath(environment_path)

        if not env_prim.IsValid():
            print(f"Warning: Environment prim {environment_path} not found")
            return False

        # Apply transform to the root environment prim
        xformable = UsdGeom.Xformable(env_prim)

        # Clear any existing transforms and set a clean translation
        # This ensures we don't accumulate transforms or create coordinate system issues
        xformable.ClearXformOpOrder()
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(offset[0], offset[1], offset[2]))

        print(f"✓ Applied clean offset {offset} to environment {environment_path}")
        return True

    def start_user_animation(self, user_id, goal_joints, duration=3.0, gripper_action=None):
        """Start animation for specific user using direct joint control First execution generates all frames headlessly,
        subsequent requests replay cached frames."""
        if user_id not in self.user_environments:
            return {"error": f"User {user_id} environment not found"}

        import numpy as np

        robot = self.user_environments[user_id]["robot"]

        if goal_joints is None:
            return {"error": "Must provide goal_joints"}

        # Convert goal_joints to numpy array
        goal_joints = np.array(goal_joints)

        # Check if we already have a complete frame cache for this animation
        cache_key = f"{user_id}_{hash(tuple(goal_joints))}_{duration}"
        if user_id in self.frame_caches and self.frame_caches[user_id].is_complete:
            # Use cached frames for instant replay
            cache = self.frame_caches[user_id]
            cache.reset_replay()
            print(f"🎬 Using cached animation for user {user_id} - {cache.frame_count} pre-generated frames")

            # Mark as active animation for the replay system
            self.active_animations[user_id] = {
                "type": "replay",
                "cache": cache,
                "start_time": time.time(),
                "active": True,
            }

            return {
                "status": "animation_started",
                "user_id": user_id,
                "mode": "cached_replay",
                "frame_count": cache.frame_count,
            }

        # First time or different animation - need to generate frames
        print(f"🎬 Generating new animation frames for user {user_id}")

        # STEP 1: Stop any existing animation for this user to ensure clean start
        if user_id in self.active_animations:
            print(f"🔄 Stopping existing animation for user {user_id} to start fresh")
            self.active_animations[user_id]["active"] = False
            del self.active_animations[user_id]

        # STEP 2: Clear any existing frame cache
        if user_id in self.frame_caches:
            self.frame_caches[user_id].clear_cache()

        # STEP 3: Create new frame cache
        fps = 30.0  # Standard frame rate
        self.frame_caches[user_id] = AnimationFrameCache(user_id, duration, fps)
        cache = self.frame_caches[user_id]

        # STEP 4: Mark user as generating frames (prevent capture requests during generation)
        self.frame_generation_in_progress.add(user_id)

        # STEP 5: CRITICAL - Reset ENTIRE environment (robot + objects) to fresh synchronized state
        print(f"🆕 PERFORMING FULL FRESH RESET for user {user_id} before animation")
        self._reset_user_environment_to_sync_state(user_id)  # ATTENTION

        # STEP 6: Get the fresh initial state from the latest synchronized state
        initial_joints = self.last_sync_config["robot_joints"]

        if gripper_action is not None:
            if gripper_action in ("grasp", "close"):
                goal_joints[-1] = 0.000
                print(f"🤏 Gripper action: {gripper_action} -> setting left gripper to closed (0.0)")
            elif gripper_action == "open":
                goal_joints[-1] = 0.044
                print(f"✋ Gripper action: {gripper_action} -> setting left gripper to open (0.044)")
            else:
                print(f"⚠️ Unknown gripper action: {gripper_action}, keeping original gripper position")

        print(f"🆕 FRESH START - Initial: {initial_joints}, Goal: {goal_joints}")

        # STEP 7: Set up CHUNKED frame generation (non-blocking)
        print(f"📹 Setting up chunked frame generation for {duration}s animation at {fps} FPS")

        try:
            cache.start_generation()

            # Set up chunked generation state
            self.chunked_generation_state[user_id] = {
                "cache": cache,
                "robot": robot,
                "initial_joints": initial_joints,
                "goal_joints": goal_joints,
                "fps": fps,
                "total_frames": cache.total_frames,
                "current_frame": 0,
                "frame_interval": 1.0 / fps,
                "generation_complete": False,
            }

            print(
                f"✅ Chunked generation setup complete for user {user_id} - will generate {cache.total_frames} frames incrementally"
            )

            # Return immediately - frames will be generated in background by worker loop
            return {
                "status": "animation_starting",
                "user_id": user_id,
                "mode": "chunked_generation_setup",
                "total_frames": cache.total_frames,
            }

        except Exception as e:
            print(f"❌ Error setting up chunked generation for user {user_id}: {e}")
            traceback.print_exc()

            # Cleanup on error
            if user_id in self.frame_caches:
                self.frame_caches[user_id].clear_cache()
                del self.frame_caches[user_id]
            if user_id in self.chunked_generation_state:
                del self.chunked_generation_state[user_id]

            return {"error": f"Animation setup failed: {str(e)}"}

        finally:
            # Always remove from generation in progress since we're now using chunked approach
            self.frame_generation_in_progress.discard(user_id)

    def stop_user_animation(self, user_id):
        """Stop animation for specific user and reset to fresh synchronized state Also cleans up frame cache to prevent
        memory leaks."""
        print(f"[Worker] 🛑 Starting stop_user_animation for user {user_id}")
        start_time = time.time()

        # CRITICAL: If this user is currently generating frames, signal immediate stop
        if user_id in self.frame_generation_in_progress or user_id in self.chunked_generation_state:
            self.animation_stop_requested.add(user_id)

        # Reset the user environment (always do this for clean state)
        reset_start = time.time()

        if user_id in self.active_animations:
            self.active_animations[user_id]["active"] = False
            del self.active_animations[user_id]

        # Reset this user's environment back to the fresh synchronized state
        self._reset_user_environment_to_sync_state(user_id)  # ATTENTION

        # Clean up frame cache when animation stops
        if user_id in self.frame_caches:
            self.frame_caches[user_id].clear_cache()
            del self.frame_caches[user_id]

        # Remove from frame generation tracking
        self.frame_generation_in_progress.discard(user_id)
        # Clear any pending stop requests
        self.animation_stop_requested.discard(user_id)

        return {"status": "animation_stopped", "user_id": user_id, "reset_to_fresh": True, "cache_cleared": True}

    def set_joint_positions_physics_inspector(self, target_positions):
        """Set joint positions using UsdPhysics.DriveAPI (direct drive targets)

        Args:
            target_positions: List of 8 target joint positions in RADIANS [joint_0 through joint_6, left_carriage_joint]
                            Function converts radians to degrees for revolute joints automatically

        """
        import math

        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()

        # Your specific joint paths and types
        joint_configs = [
            ("/World/wxai/joints/joint_0", "angular"),  # revolute
            ("/World/wxai/joints/joint_1", "angular"),  # revolute
            ("/World/wxai/joints/joint_2", "angular"),  # revolute
            ("/World/wxai/joints/joint_3", "angular"),  # revolute
            ("/World/wxai/joints/joint_4", "angular"),  # revolute
            ("/World/wxai/joints/joint_5", "angular"),  # revolute
            ("/World/wxai/joints/left_carriage_joint", "linear"),  # prismatic
        ]

        # Set drive targets directly
        for i, (joint_path, drive_type) in enumerate(joint_configs[: len(target_positions)]):
            joint_prim = stage.GetPrimAtPath(joint_path)
            if joint_prim:
                drive = UsdPhysics.DriveAPI.Get(joint_prim, drive_type)
                if drive:
                    if drive_type == "angular":
                        # Convert radians to degrees for USD DriveAPI
                        target_degrees = math.degrees(target_positions[i])
                        drive.GetTargetPositionAttr().Set(target_degrees)
                    else:  # linear/prismatic
                        # Use meters directly
                        drive.GetTargetPositionAttr().Set(float(target_positions[i]))
                else:
                    print(f"Warning: No {drive_type} drive found for {joint_path}")
            else:
                print(f"Warning: Joint not found at {joint_path}")

        print(f"✅ Set {len(target_positions)} joint drive targets (DriveAPI method)")

    def process_chunked_frame_generation(self, frames_per_chunk=3):
        """Generate a few frames per call.

        Drives only DOFs 0..6 (arm + left finger).
        The right finger (index 7) is a USD mimic and is never commanded here.

        """
        import numpy as np
        from omni.isaac.core.utils.types import ArticulationAction

        work_done = False
        INDICES_7 = list(range(7))  # arm + left finger only

        for user_id in list(self.chunked_generation_state.keys()):
            state = self.chunked_generation_state[user_id]

            # Stop request?
            if user_id in self.animation_stop_requested:
                self.animation_stop_requested.discard(user_id)
                if user_id in self.frame_caches:
                    self.frame_caches[user_id].clear_cache()
                    del self.frame_caches[user_id]
                del self.chunked_generation_state[user_id]
                continue

            if state.get("generation_complete"):
                continue

            cache = state["cache"]
            robot = state["robot"]
            frames_processed = 0

            while frames_processed < frames_per_chunk and state["current_frame"] < state["total_frames"]:
                f = state["current_frame"]

                # One-time trajectory setup (7-DOF only)
                if f == 0:
                    q0_full = robot.get_joint_positions()  # likely 8-long
                    q0_7 = np.array(q0_full[:7], dtype=np.float32)  # use first 7
                    qg = np.array(state["goal_joints"], dtype=np.float32)
                    if qg.shape[0] == 8:
                        qg = qg[:7]
                    elif qg.shape[0] != 7:
                        raise ValueError(f"goal_joints must be 7 (or 8 to be sliced). Got {qg.shape[0]}")
                    state["q0_7"] = q0_7
                    state["qg_7"] = qg
                    state["T"] = state["total_frames"] / state["fps"]

                q0_7 = state["q0_7"]
                right0 = q0_7[self.GRIPPER_LEFT_IDX]
                q0_8 = q0_7.tolist() + [right0]

                robot.apply_action(
                    ArticulationAction(joint_positions=q0_8, joint_velocities=[0.0] * 8)  # freeze all 8 DOFs
                )

                # Min-jerk interpolation on 7 DOFs
                T = state["T"]
                tau = 0.0 if T <= 0 else max(0.0, min((f / state["fps"]) / T, 1.0))
                s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
                sdot = 0.0 if T <= 0 else (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T

                q0_7, qg_7 = state["q0_7"], state["qg_7"]
                q_des_7 = q0_7 + s * (qg_7 - q0_7)
                qd_des_7 = sdot * (qg_7 - q0_7)

                # Command ONLY 0..6; never the mimic DOF
                right_pos = q_des_7[self.GRIPPER_LEFT_IDX]
                right_vel = qd_des_7[self.GRIPPER_LEFT_IDX]

                q_des_8 = q_des_7.tolist() + [right_pos]
                qd_des_8 = qd_des_7.tolist() + [right_vel]

                robot.apply_action(
                    ArticulationAction(
                        joint_positions=q_des_7,
                        joint_velocities=qd_des_7,
                        joint_indices=list(range(7)),
                        # no joint_indices → applies to all 8 DOFs
                    )
                )

                # Step + capture
                self.world.step(render=True)
                frame_data = self._capture_user_frame_to_cache(user_id, f)
                if frame_data:
                    cache.add_frame(f, frame_data)

                state["current_frame"] += 1
                frames_processed += 1
                work_done = True

            # Done with this user's sequence?
            if state["current_frame"] >= state["total_frames"]:
                state["generation_complete"] = True
                cache.reset_replay()
                self.active_animations[user_id] = {
                    "type": "replay",
                    "cache": cache,
                    "start_time": time.time(),
                    "active": True,
                }
                del self.chunked_generation_state[user_id]

        return work_done

    def _check_for_stop_command_during_generation(self, user_id: int) -> bool:
        """Check if a stop command is pending for this user during frame generation Returns True if stop command
        detected, False otherwise."""
        if not self.worker_communication_dir:
            return False  # Fallback to flag-based checking if no communication dir set

        try:
            import json

            command_file = f"{self.worker_communication_dir}/commands.json"
            command_signal_file = f"{command_file}.signal"

            # Check if there's a pending command
            if os.path.exists(command_signal_file) and os.path.exists(command_file):
                try:
                    # Check if file is not empty before trying to parse JSON
                    if os.path.getsize(command_file) > 0:
                        with open(command_file, "r") as f:
                            content = f.read().strip()
                            if content:  # Only parse if there's actual content
                                command = json.loads(content)

                                # Check if it's a stop command for this user
                                if command.get("action") == "stop_user_animation" and command.get("user_id") == user_id:
                                    print(f"🛑 DETECTED stop command for user {user_id} during frame generation!")
                                    return True
                    else:
                        print(f"Warning: Command file {command_file} is empty")
                except json.JSONDecodeError as json_err:
                    print(f"Warning: Invalid JSON in command file {command_file}: {json_err}")
                except Exception as read_err:
                    print(f"Warning: Could not read command file {command_file}: {read_err}")

        except Exception as e:
            # If file reading fails, continue with generation
            print(f"Warning: Could not check for stop commands: {e}")

        return False

    def _capture_user_frame_to_cache(self, user_id: int, frame_index: int) -> dict | None:
        """Capture a single frame for the frame cache during generation Returns dict of {camera_name: image_path} or
        None if failed."""
        if user_id not in self.user_environments:
            return None

        import os

        # Create frame-specific directory
        frame_dir = f"/tmp/isaac_worker/user_{user_id}_frames/frame_{frame_index:04d}"
        os.makedirs(frame_dir, exist_ok=True)

        cameras = self.user_environments[user_id]["cameras"]
        captured_files = {}

        for camera_name, camera in cameras.items():
            try:
                rgb_data = camera.get_rgb()
                if rgb_data is not None:
                    clean_name = camera_name.lower().replace("camera_", "")
                    filename = f"{clean_name}_{frame_index:04d}.jpg"
                    filepath = os.path.join(frame_dir, filename)
                    Image.fromarray(rgb_data).save(filepath, "JPEG", quality=85)
                    captured_files[clean_name] = filepath
            except Exception as e:
                print(f"Warning: Failed to capture {camera_name} for frame {frame_index}: {e}")

        return captured_files if captured_files else None

    def cleanup_all_frame_caches(self):
        """Clean up all frame caches - call this on shutdown"""
        print("🧹 Cleaning up all frame caches...")
        for user_id in list(self.frame_caches.keys()):
            try:
                self.frame_caches[user_id].clear_cache()
                del self.frame_caches[user_id]
                print(f"✅ Cleared cache for user {user_id}")
            except Exception as e:
                print(f"⚠️ Error clearing cache for user {user_id}: {e}")

        self.frame_caches.clear()
        self.frame_generation_in_progress.clear()
        print("✅ All frame caches cleaned up")

    def _reset_user_environment_to_sync_state(self, user_id):
        """Reset a user's environment back to the last synchronized state.

        - User 0: Resets robot + objects using stored object references
        - Cloned users: Resets robot + objects using scene registry objects
        Both approaches ensure fresh initial state for animation restart

        """
        if user_id not in self.user_environments:
            return

        import numpy as np
        import omni.usd
        from pxr import Gf, UsdGeom

        try:
            self.robot.set_joint_positions(
                np.zeros(8, dtype=float),
            )
            env_data = self.user_environments[user_id]
            robot = env_data["robot"]
            # Use the correct object references for each environment type

            object_states = self.last_sync_config.get(
                "object_poses",
                {
                    "Cube_Blue": {"pos": [0.6, 0.0, 0.1], "rot": [0, 0, 0, 1]},
                    "Cube_Red": {"pos": [0.6, 0.2, 0.1], "rot": [0, 0, 0, 1]},
                    "Tennis": {"pos": [0.6, -0.2, 0.1], "rot": [0, 0, 0, 1]},
                },
            )

            if user_id == 0:
                # User 0: Use original object references - reset to absolute positions

                if "Cube_Blue" in object_states and "Cube_Blue" in self.objects:
                    state = object_states["Cube_Blue"]
                    if state is not None:  # Skip if pose estimation failed
                        pos = np.array(state["pos"])
                        rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])
                        if self.objects["Cube_Blue"].is_valid():
                            # Reset position AND clear physics state
                            self.objects["Cube_Blue"].set_world_pose(position=pos, orientation=rot)
                            self.objects["Cube_Blue"].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                            self.objects["Cube_Blue"].set_angular_velocity(np.array([0.0, 0.0, 0.0]))

                if "Cube_Red" in object_states and "Cube_Red" in self.objects:
                    state = object_states["Cube_Red"]
                    if state is not None:  # Skip if pose estimation failed
                        pos = np.array(state["pos"])
                        rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])
                        if self.objects["Cube_Red"].is_valid():
                            # Reset position AND clear physics state
                            self.objects["Cube_Red"].set_world_pose(position=pos, orientation=rot)
                            self.objects["Cube_Red"].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                            self.objects["Cube_Red"].set_angular_velocity(np.array([0.0, 0.0, 0.0]))

                if "Tennis" in object_states and "tennis_ball" in self.objects:
                    state = object_states["Tennis"]
                    if state is not None:  # Skip if pose estimation failed
                        pos = np.array(state["pos"])
                        rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])
                        if self.objects["tennis_ball"].is_valid():
                            # Reset position AND clear physics state - CRITICAL for sphere/ball objects
                            self.objects["tennis_ball"].set_world_pose(position=pos, orientation=rot)
                            self.objects["tennis_ball"].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                            self.objects["tennis_ball"].set_angular_velocity(np.array([0.0, 0.0, 0.0]))

            else:
                # Cloned environments: Use scene registry objects WITH SPATIAL OFFSET

                # Calculate the spatial offset for this environment (same as sync logic)
                environment_spacing = 50.0
                spatial_offset = np.array([user_id * environment_spacing, user_id * environment_spacing, 0])

                # Reset each object using scene registry with spatial offset
                object_mappings = [
                    ("Cube_Blue", f"Cube_Blue_user_{user_id}"),
                    ("Cube_Red", f"Cube_Red_user_{user_id}"),
                    ("Tennis", f"tennis_user_{user_id}"),
                ]

                for config_key, scene_name in object_mappings:
                    if config_key in object_states:
                        state = object_states[config_key]
                        if state is None:  # Skip if pose estimation failed
                            continue

                        # CRITICAL: Apply spatial offset to object position (same as sync)
                        original_pos = np.array(state["pos"])
                        offset_pos = original_pos + spatial_offset
                        rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])

                        # Try to get object from scene registry
                        if self.world.scene.object_exists(scene_name):
                            scene_obj = self.world.scene.get_object(scene_name)
                            # Reset position AND clear physics state
                            scene_obj.set_world_pose(position=offset_pos, orientation=rot)
                            scene_obj.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                            scene_obj.set_angular_velocity(np.array([0.0, 0.0, 0.0]))

            for step in range(50):
                self.world.step(render=True)

            # robot.initialize()

            self.set_robot_joints()

            # Let physics settle after all resets are complete (extended for gripper operations)
            # for step in range(12):  # Increased from 8 to account for gripper operations
            #     self.world.step(render=True)

        except Exception as e:
            print(f"❌ Failed to reset user {user_id} environment: {e}")

    def update_animations(self):
        """Update all active animations (called each frame) Uses efficient replay system with pre-generated frames."""
        import numpy as np

        current_time = time.time()

        for user_id, anim_data in list(self.active_animations.items()):
            if not anim_data["active"]:
                continue

            # Frame-based replay system - no physics simulation needed
            # Frames are already generated and cached, just update timing
            cache = anim_data["cache"]
            if cache.is_complete:
                # The cache handles its own timing and looping
                # No robot position updates needed as frames contain all visual data
                pass
            else:
                # If cache is not complete, something went wrong
                print(f"⚠️ Animation cache for user {user_id} is not complete, stopping animation")
                anim_data["active"] = False

    def capture_user_frame(self, user_id, output_dir):
        """Capture current frame for specific user Serves cached frames if available, falls back to live capture."""
        if user_id not in self.user_environments:
            return None

        import os

        # Check if this user has a frame cache and active animation
        if (
            user_id in self.active_animations
            and self.active_animations[user_id].get("active")
            and user_id in self.frame_caches
        ):

            cache = self.frame_caches[user_id]
            current_frame_data = cache.get_current_replay_frame()

            if current_frame_data:
                # Serve cached frame files directly (no copying - just return the cached paths)
                captured_files = {}

                for camera_name, cached_filepath in current_frame_data.items():
                    try:
                        if os.path.exists(cached_filepath):
                            # Return the cached file path directly - no copying needed
                            captured_files[camera_name] = cached_filepath
                        else:
                            print(f"⚠️ Cached frame file not found: {cached_filepath}")
                    except Exception as e:
                        print(f"⚠️ Error accessing cached frame {cached_filepath}: {e}")

                if captured_files:
                    # Return cached frame data directly (no file copying)
                    return captured_files
                else:
                    print(f"⚠️ Failed to serve cached frames for user {user_id}, falling back to live capture")

        # FALLBACK: Original live capture system (for non-cached animations or failures)
        os.makedirs(f"{output_dir}/user_{user_id}", exist_ok=True)

        cameras = self.user_environments[user_id]["cameras"]
        captured_files = {}

        for camera_name, camera in cameras.items():
            rgb_data = camera.get_rgb()
            if rgb_data is not None:
                clean_name = camera_name.lower().replace("camera_", "")
                filename = f"user_{user_id}_{clean_name}.jpg"
                filepath = f"{output_dir}/user_{user_id}/{filename}"
                Image.fromarray(rgb_data).save(filepath, "JPEG", quality=85)
                captured_files[clean_name] = filepath

        return captured_files

    def animation_loop(self, output_dir):
        """Main animation loop for streaming frames."""
        print("Starting animation loop...")

        frame_count = 0
        last_capture_time = time.time()
        capture_interval = 1.0 / 30.0  # 30 FPS

        while self.running and self.animation_mode:
            # 1) advance generation (this produces actions + steps caches forward)
            self.process_chunked_frame_generation(frames_per_chunk=3)

            # 2) advance physics
            self.world.step(render=True)

            # 3) update replays
            self.update_animations()

            # 4) capture frames at intervals
            current_time = time.time()
            if current_time - last_capture_time >= capture_interval:
                for user_id in self.active_animations:
                    self.capture_user_frame(user_id, output_dir)
                last_capture_time = current_time

            frame_count += 1

            # Small delay to prevent 100% CPU usage
            time.sleep(0.001)

        print("Animation loop ended")

    def handle_command(self, command):
        """Handle runtime commands from backend."""
        action = command.get("action")

        if action == "start_animation":
            return self.start_user_animation(
                user_id=command["user_id"],
                goal_joints=command.get("goal_joints"),
                duration=command.get("duration", 3.0),
            )

        elif action == "stop_animation":
            return self.stop_user_animation(command["user_id"])

        elif action == "terminate":
            # Clean up frame caches before terminating
            self.cleanup_all_frame_caches()
            self.running = False
            return {"status": "terminating", "caches_cleaned": True}

        else:
            return {"error": f"Unknown action: {action}"}
