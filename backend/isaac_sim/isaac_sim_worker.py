#!/usr/bin/env python3
"""
Isaac Sim worker script with two modes:
1. Static image capture (for frontend interaction)
2. Animation mode (for physics simulation with direct joint control)
"""

import os
import sys
import json
import argparse
import time
import threading
import signal
import traceback
from collections import defaultdict
from PIL import Image
from isaacsim import SimulationApp

class AnimationFrameCache:
    """Cache system for storing and replaying animation frames efficiently"""
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
        """Mark the start of frame generation"""
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
                print(f"✅ Frame generation complete for user {self.user_id}: {self.frame_count} frames in {generation_time:.2f}s")
            else:
                print(f"✅ Frame generation complete for user {self.user_id}: {self.frame_count} frames (no timing available)")
        
    def get_current_replay_frame(self) -> dict | None:
        """Get the current frame for replay based on elapsed time"""
        if not self.is_complete:
            return None
            
        if self.replay_start_time is None:
            self.replay_start_time = time.time()
            self.current_replay_frame = 0
            
        # Calculate which frame we should be showing
        elapsed = time.time() - self.replay_start_time
        target_frame = int((elapsed % self.duration) * self.fps)
        
        # If we've looped, reset the replay start time for smoother looping
        if target_frame < self.current_replay_frame:
            self.replay_start_time = time.time()
            target_frame = 0
            
        self.current_replay_frame = target_frame
        
        # Return the frame data
        return self.frames.get(target_frame)
        
    def reset_replay(self):
        """Reset replay to start from beginning"""
        self.replay_start_time = None
        self.current_replay_frame = 0
        
    def clear_cache(self):
        """Clear all cached frames and clean up files"""
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
        
        # === NEW: Frame cache system for efficient animation replay ===
        self.frame_caches = {}  # user_id -> AnimationFrameCache
        self.frame_generation_in_progress = set()  # Track which users are generating frames
        self.animation_stop_requested = set()  # Track users for whom stop has been requested during generation
        self.worker_communication_dir = None  # Will be set by persistent worker for direct command checking
        
        # === NEW: Chunked frame generation system ===
        self.chunked_generation_state = {}  # user_id -> generation state for async processing
        
    def initialize_simulation(self, config):
        """One-time simulation setup that can be reused across state updates"""
        if self.simulation_initialized:
            return
            
        import numpy as np
        import carb
        import omni.usd
        from omni.isaac.core import World
        from omni.isaac.core.prims import RigidPrim
        from omni.isaac.core.articulations import Articulation
        from pxr import Gf, UsdGeom, Usd
        from isaacsim.sensors.camera import Camera, get_all_camera_objects
        
        # Configuration
        USD_PATH = config['usd_path']
        ROBOT_PATH = "/World/wxai"
        OBJ_CUBE_01_PATH = "/World/Cube_01"
        OBJ_CUBE_02_PATH = "/World/Cube_02"
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
                simulation_app = getattr(sys.modules[__name__], 'simulation_app', None)
                if simulation_app:
                    simulation_app.update()

        # Create the World object (only once)
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        print("Stage loaded and World object created.")

        # Get handles to the prims (store for reuse)
        self.robot = self.world.scene.add(Articulation(prim_path=ROBOT_PATH, name="widowx_robot"))
        self.objects['cube_01'] = self.world.scene.add(RigidPrim(prim_path=OBJ_CUBE_01_PATH, name="cube_01"))
        self.objects['cube_02'] = self.world.scene.add(RigidPrim(prim_path=OBJ_CUBE_02_PATH, name="cube_02"))
        self.objects['tennis_ball'] = self.world.scene.add(RigidPrim(prim_path=OBJ_TENNIS_PATH, name="tennis_ball"))

        # Get cameras (only once)
        stage = omni.usd.get_context().get_stage()
        all_cameras = get_all_camera_objects(root_prim='/')
        self.cameras = {all_cameras[i].name: all_cameras[i] for i in range(len(all_cameras))}

        # Reset world and initialize cameras (only once)
        self.world.reset()

        # Initialize cameras (only once)
        for camera in all_cameras:
            camera.initialize()
            camera.set_resolution((640,480))
            camera.add_rgb_to_frame()
            
        # Create robot hide/show functions (only once)
        def hide_robot():
            """Alternative: Move robot far away (preserves everything)"""
            hide_robot.original_pos, hide_robot.original_rot = self.robot.get_world_pose()
            hide_robot.current_joint_positions = self.robot.get_joint_positions()
            # Move robot below ground
            self.robot.set_world_pose(position=np.array([0, 0, -100]), orientation=hide_robot.original_rot)

        def show_robot():
            """Restore robot to original position"""
            self.robot.set_world_pose(position=hide_robot.original_pos, 
                                    orientation=hide_robot.original_rot)
            self.robot.set_joint_positions(hide_robot.current_joint_positions)
        
        self.hide_robot_funcs = {'hide': hide_robot, 'show': show_robot}
        
        self.simulation_initialized = True
        print("Simulation initialized successfully - ready for state updates")
        
    def update_state(self, config):
        """Update robot joints and object poses without recreating simulation"""
        import numpy as np
        
        if not self.simulation_initialized:
            raise RuntimeError("Must call initialize_simulation() first")
            
        # Store the config for animation reset purposes
        self.last_sync_config = config.copy()
            
        # Get state from config
        initial_q = np.array(config.get('robot_joints', [0.0] * 7))
        initial_q = np.append(initial_q, initial_q[-1])
        object_states = config.get('object_poses', {
            "Cube_01": {"pos": [0.5, 0.0, 0.1], "rot": [0, 0, 0, 1]},
            "Cube_02": {"pos": [0.5, 0.2, 0.1], "rot": [0, 0, 0, 1]},
            "Tennis": {"pos": [0.5, -0.2, 0.1], "rot": [0, 0, 0, 1]}
        })

        # Grasp detection: Check if gripper force exceeds threshold (same as frontend)
        GRASP_THRESHOLD_N = 50.0  # Same threshold as frontend
        gripper_force = config.get('left_carriage_external_force', 0.0)
        
        if abs(gripper_force) >= GRASP_THRESHOLD_N:
            # Object is grasped - set gripper joints to closed position
            initial_q[-2] = 0.0  # 7th dimension (left gripper)
            initial_q[-1] = 0.0  # 8th dimension (right gripper)
            print(f"🤏 Grasp detected (force: {gripper_force:.1f}N >= {GRASP_THRESHOLD_N}N) - closing gripper in simulation")
        else:
            print(f"✋ No grasp (force: {gripper_force:.1f}N < {GRASP_THRESHOLD_N}N) - using original gripper positions")

        self.last_sync_config['robot_joints'] = initial_q

        # Update robot joints
        self.robot.set_joint_positions(initial_q)

        # Update object poses
        if self.objects['cube_01'].is_valid():
            state = object_states.get("Cube_01")
            if state:
                pos = np.array(state["pos"])
                rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])
                self.objects['cube_01'].set_world_pose(position=pos, orientation=rot)

        if self.objects['cube_02'].is_valid():
            state = object_states.get("Cube_02")
            if state:
                pos = np.array(state["pos"])
                rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])
                self.objects['cube_02'].set_world_pose(position=pos, orientation=rot)
                
        if self.objects['tennis_ball'].is_valid():
            state = object_states.get("Tennis")
            if state:
                pos = np.array(state["pos"])
                rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])
                self.objects['tennis_ball'].set_world_pose(position=pos, orientation=rot)

        # Let physics settle
        for step in range(5):
            self.world.step(render=True)
            
        print("State updated successfully")
        
    def capture_current_state_images(self, output_dir):
        """Capture images with current state (robot temporarily hidden for capture only)"""
        import os
        from PIL import Image
        
        if not self.simulation_initialized:
            raise RuntimeError("Must call initialize_simulation() first")
            
        os.makedirs(output_dir, exist_ok=True)
        
        print("Capturing images with current state...")
        
        # Temporarily hide robot for static capture
        self.hide_robot_funcs['hide']()
        
        # Let physics settle with robot hidden
        for step in range(10):
            self.world.step(render=True)

        # Capture static images
        front_rgb = self.cameras['Camera_Front'].get_rgb()
        left_rgb = self.cameras['Camera_Left'].get_rgb()
        right_rgb = self.cameras['Camera_Right'].get_rgb()
        top_rgb = self.cameras['Camera_Top'].get_rgb()

        # IMPORTANT: Restore robot visibility after capture - environment should always have robot visible
        self.hide_robot_funcs['show']()
        
        # Let physics settle with robot restored
        for step in range(5):
            self.world.step(render=True)

        # Save static images
        Image.fromarray(front_rgb).save(f'{output_dir}/static_front_image.jpg', 'JPEG', quality=90)
        Image.fromarray(left_rgb).save(f'{output_dir}/static_left_image.jpg', 'JPEG', quality=90)
        Image.fromarray(right_rgb).save(f'{output_dir}/static_right_image.jpg', 'JPEG', quality=90)
        Image.fromarray(top_rgb).save(f'{output_dir}/static_top_image.jpg', 'JPEG', quality=90)

        return {
            "front_rgb": f'{output_dir}/static_front_image.jpg',
            "left_rgb": f'{output_dir}/static_left_image.jpg',
            "right_rgb": f'{output_dir}/static_right_image.jpg',
            "top_rgb": f'{output_dir}/static_top_image.jpg',
            "status": "static_images_captured"
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
            from omni.isaac.cloner import Cloner
            import numpy as np
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
                self.hide_robot_funcs['show']()
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
                    offset = [user_id * environment_spacing, user_id * environment_spacing, 0]  # X and Y axis offset for diagonal spacing
                    
                    print(f"User {user_id}: Attempting minimal clone at offset {offset}")
                    
                    try:
                        # Clone with very simple approach
                        cloner.clone(
                            source_prim_path="/World",
                            prim_paths=[target_path],
                            positions=[np.array(offset)],
                            copy_from_source=True
                        )
                        
                        # Let physics settle
                        for step in range(20):
                            self.world.step(render=True)
                        
                        # Get robot
                        robot_path = f"{target_path}/wxai"
                        user_robot = self.world.scene.add(Articulation(
                            prim_path=robot_path, 
                            name=f"robot_user_{user_id}"
                        ))
                        
                        # Register cloned objects in scene registry for easy access
                        from omni.isaac.core.prims import RigidPrim
                        try:
                            cube_01_path = f"{target_path}/Cube_01"
                            cube_02_path = f"{target_path}/Cube_02" 
                            tennis_path = f"{target_path}/Tennis"
                            
                            self.world.scene.add(RigidPrim(prim_path=cube_01_path, name=f"cube_01_user_{user_id}"))
                            self.world.scene.add(RigidPrim(prim_path=cube_02_path, name=f"cube_02_user_{user_id}"))
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
                    'robot': user_robot,
                    'cameras': user_cameras,
                    'world_path': target_path,
                    'spatial_offset': [0, 0, 0]
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
        """Synchronize all animation environments to match the current state"""
        if not self.animation_mode:
            print("Animation mode not initialized, skipping sync")
            return
            
        import numpy as np
        import omni.usd
        from pxr import Gf, UsdGeom
            
        print("Synchronizing animation environments to new state...")
        
        # Store the sync config so animations can reset to this state
        self.last_sync_config = config.copy()
        
        # Get state from config
        initial_q = np.array(config.get('robot_joints', [0.0] * 7))
        initial_q = np.append(initial_q, initial_q[-1])
        object_states = config.get('object_poses', {
            "Cube_01": {"pos": [0.5, 0.0, 0.1], "rot": [0, 0, 0, 1]},
            "Cube_02": {"pos": [0.5, 0.2, 0.1], "rot": [0, 0, 0, 1]},
            "Tennis": {"pos": [0.5, -0.2, 0.1], "rot": [0, 0, 0, 1]}
        })
        
        # Update each animation environment
        for user_id, env_data in self.user_environments.items():
            try:
                robot = env_data['robot']
                world_path = env_data['world_path']
                
                # Update robot joints
                robot.set_joint_positions(initial_q)
                
                # Update object poses for ALL environments using appropriate object references
                if user_id == 0:
                    # User 0: Use original object references - reset to absolute positions
                    print(f"🔧 Syncing objects for user 0 (original environment)")
                    
                    if 'Cube_01' in object_states:
                        state = object_states['Cube_01']
                        pos = np.array(state["pos"])
                        rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])
                        self.objects['cube_01'].set_world_pose(position=pos, orientation=rot)
                        self.objects['cube_01'].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                        self.objects['cube_01'].set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                        print(f"✅ Synced Cube_01 to {pos} (physics cleared)")
                        
                    if 'Cube_02' in object_states:
                        state = object_states['Cube_02'] 
                        pos = np.array(state["pos"])
                        rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])
                        self.objects['cube_02'].set_world_pose(position=pos, orientation=rot)
                        self.objects['cube_02'].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                        self.objects['cube_02'].set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                        print(f"✅ Synced Cube_02 to {pos} (physics cleared)")
                        
                    if 'Tennis' in object_states:
                        state = object_states['Tennis']
                        pos = np.array(state["pos"])
                        rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])
                        self.objects['tennis_ball'].set_world_pose(position=pos, orientation=rot)
                        self.objects['tennis_ball'].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                        self.objects['tennis_ball'].set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                        print(f"✅ Synced Tennis to {pos} (physics cleared)")
                else:
                    # Cloned environments: Sync objects using scene registry WITH SPATIAL OFFSET
                    print(f"🔧 Syncing objects for user {user_id} (cloned environment) using scene registry WITH OFFSET")
                    
                    # Calculate the spatial offset for this environment
                    # User environments are spaced diagonally: [user_id * 50, user_id * 50, 0]
                    environment_spacing = 50.0
                    spatial_offset = np.array([user_id * environment_spacing, user_id * environment_spacing, 0])
                    print(f"📍 User {user_id} spatial offset: {spatial_offset}")
                    
                    # Sync each cloned object using scene registry with spatial offset applied
                    object_mappings = [
                        ("Cube_01", f"cube_01_user_{user_id}"),
                        ("Cube_02", f"cube_02_user_{user_id}"),
                        ("Tennis", f"tennis_user_{user_id}")
                    ]
                    
                    for config_key, scene_name in object_mappings:
                        if config_key in object_states:
                            state = object_states[config_key]
                            
                            # CRITICAL: Apply spatial offset to object position
                            original_pos = np.array(state["pos"])
                            offset_pos = original_pos + spatial_offset
                            rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])
                            
                            print(f"🔄 {config_key}: Original pos {original_pos} + offset {spatial_offset} = {offset_pos}")
                            
                            # Set object position using scene registry with offset
                            if self.world.scene.object_exists(scene_name):
                                scene_obj = self.world.scene.get_object(scene_name)
                                scene_obj.set_world_pose(position=offset_pos, orientation=rot)
                                scene_obj.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                                scene_obj.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                                print(f"✅ Synced {config_key} via scene registry to offset pos: {offset_pos} (physics cleared)")
                            else:
                                print(f"⚠️ Scene object {scene_name} not found for user {user_id} - skipping sync")
                                
                    print(f"📍 User {user_id} objects synced using scene registry WITH SPATIAL OFFSET")
                    
                print(f"Synced environment {user_id} ({world_path})")
                
            except Exception as e:
                print(f"Warning: Failed to sync environment {user_id}: {e}")
        
        # Let physics settle across all environments
        for step in range(5):
            self.world.step(render=True)
            
        print("Animation environment synchronization complete")

    def _update_object_transform(self, stage, object_path, position, rotation):
        """Helper method to update object transform with proper USD precision handling"""
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
            if 'quatd' in str(type_name):
                # Double precision quaternion
                quat = Gf.Quatd(rotation[3], rotation[0], rotation[1], rotation[2])  # Gf.Quatd(w, x, y, z)
            else:
                # Float precision quaternion (default)
                quat = Gf.Quatf(rotation[3], rotation[0], rotation[1], rotation[2])  # Gf.Quatf(w, x, y, z)
            
            orient_op.Set(quat)

    def _apply_environment_offset(self, environment_path, offset):
        """Apply spatial offset to an entire cloned environment"""
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
        """Start animation for specific user using direct joint control
        NEW: First execution generates all frames headlessly, subsequent requests replay cached frames"""
        if user_id not in self.user_environments:
            return {"error": f"User {user_id} environment not found"}
            
        import numpy as np
        
        robot = self.user_environments[user_id]['robot']
        
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
                'type': 'replay',
                'cache': cache,
                'start_time': time.time(),
                'active': True
            }
            
            return {
                "status": "animation_started", 
                "user_id": user_id, 
                "mode": "cached_replay",
                "frame_count": cache.frame_count
            }
        
        # First time or different animation - need to generate frames
        print(f"🎬 Generating new animation frames for user {user_id}")
        
        # STEP 1: Stop any existing animation for this user to ensure clean start
        if user_id in self.active_animations:
            print(f"🔄 Stopping existing animation for user {user_id} to start fresh")
            self.active_animations[user_id]['active'] = False
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
        self._reset_user_environment_to_sync_state(user_id)
        
        # STEP 6: Get the fresh initial state from the latest synchronized state
        if hasattr(self, 'last_sync_config') and 'robot_joints' in self.last_sync_config:
            initial_joints_7d = np.array(self.last_sync_config['robot_joints'])
            print(f"📍 Using synchronized initial state: {initial_joints_7d}")
        else:
            # Fallback to default position if no sync config available
            initial_joints_7d = np.array([0.0] * 7)
            print("⚠️ No sync config available, using default initial position")
        
        # Convert to 8D (add mimic joint)
        initial_joints = np.append(initial_joints_7d, initial_joints_7d[-1])
        
        # Ensure goal_joints has same dimension as initial_joints
        if len(goal_joints) == 7:
            goal_joints = np.append(goal_joints, goal_joints[-1])
            print(f"Extended goal_joints from 7 to 8 dimensions: {goal_joints}")
        elif len(goal_joints) != 8:
            self.frame_generation_in_progress.discard(user_id)
            return {"error": f"Goal joints dimension mismatch: got {len(goal_joints)}, expected 7 or 8"}
            
        # Handle gripper action if provided
        if gripper_action is not None:
            if gripper_action == "grasp" or gripper_action == "close":
                goal_joints[-1] = goal_joints[-2] = 0.000  # Closed gripper position
                print(f"🤏 Gripper action: {gripper_action} -> setting gripper to closed (0.0)")
            elif gripper_action == "open":
                goal_joints[-1] = goal_joints[-2] = 0.044  # Open gripper position  
                print(f"✋ Gripper action: {gripper_action} -> setting gripper to open (0.044)")
            else:
                print(f"⚠️ Unknown gripper action: {gripper_action}, keeping original gripper position")
            
        print(f"🆕 FRESH START - Initial: {initial_joints}, Goal: {goal_joints}")
        
        # STEP 7: Set up CHUNKED frame generation (non-blocking)
        print(f"📹 Setting up chunked frame generation for {duration}s animation at {fps} FPS")
        
        try:
            cache.start_generation()
            
            # Set up chunked generation state
            self.chunked_generation_state[user_id] = {
                'cache': cache,
                'robot': robot,
                'initial_joints': initial_joints,
                'goal_joints': goal_joints,
                'fps': fps,
                'total_frames': cache.total_frames,
                'current_frame': 0,
                'frame_interval': 1.0 / fps,
                'generation_complete': False
            }
            
            print(f"✅ Chunked generation setup complete for user {user_id} - will generate {cache.total_frames} frames incrementally")
            
            # Return immediately - frames will be generated in background by worker loop
            return {
                "status": "animation_starting", 
                "user_id": user_id, 
                "mode": "chunked_generation_setup",
                "total_frames": cache.total_frames
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
        """Stop animation for specific user and reset to fresh synchronized state
        NEW: Also cleans up frame cache to prevent memory leaks"""
        print(f"[Worker] 🛑 Starting stop_user_animation for user {user_id}")
        start_time = time.time()
        
        # CRITICAL: If this user is currently generating frames, signal immediate stop
        if user_id in self.frame_generation_in_progress or user_id in self.chunked_generation_state:
            self.animation_stop_requested.add(user_id)
        
        # Reset the user environment (always do this for clean state)
        reset_start = time.time()
        
        if user_id in self.active_animations:
            self.active_animations[user_id]['active'] = False
            del self.active_animations[user_id]
            
        # Reset this user's environment back to the fresh synchronized state
        self._reset_user_environment_to_sync_state(user_id)
            
        # NEW: Clean up frame cache when animation stops
        if user_id in self.frame_caches:
            self.frame_caches[user_id].clear_cache()
            del self.frame_caches[user_id]
        
        # Remove from frame generation tracking
        self.frame_generation_in_progress.discard(user_id)
        # Clear any pending stop requests
        self.animation_stop_requested.discard(user_id)
            
        return {"status": "animation_stopped", "user_id": user_id, "reset_to_fresh": True, "cache_cleared": True}
        
    def process_chunked_frame_generation(self, frames_per_chunk=3):
        """Process a few frames of chunked generation for all active users
        Returns True if any generation work was done, False if all complete"""
        import numpy as np
        work_done = False
        
        for user_id in list(self.chunked_generation_state.keys()):
            state = self.chunked_generation_state[user_id]
            
            # Check if this user's generation should be stopped
            if user_id in self.animation_stop_requested:
                print(f"🛑 Stopping chunked generation for user {user_id} due to stop request")
                self.animation_stop_requested.discard(user_id)
                # Clean up
                if user_id in self.frame_caches:
                    self.frame_caches[user_id].clear_cache()
                    del self.frame_caches[user_id]
                del self.chunked_generation_state[user_id]
                continue
                
            if state['generation_complete']:
                continue
                
            # Process a chunk of frames
            cache = state['cache']
            robot = state['robot']
            frames_processed = 0
            
            while (frames_processed < frames_per_chunk and 
                   state['current_frame'] < state['total_frames']):
                
                frame_idx = state['current_frame']
                
                # Calculate progress (0.0 to 1.0)
                progress = frame_idx / (state['total_frames'] - 1) if state['total_frames'] > 1 else 0.0
                
                # Smooth interpolation (ease-in-out)
                smooth_progress = 0.5 * (1 - np.cos(np.pi * progress))
                
                # Interpolate joint positions
                current_joints = state['initial_joints'] + (state['goal_joints'] - state['initial_joints']) * smooth_progress
                
                # Apply to robot
                robot.set_joint_positions(current_joints)
                
                # Let physics settle
                self.world.step(render=True)
                
                # Capture frame to cache
                frame_data = self._capture_user_frame_to_cache(user_id, frame_idx)
                if frame_data:
                    cache.add_frame(frame_idx, frame_data)
                    
                state['current_frame'] += 1
                frames_processed += 1
                work_done = True
                
                # Progress indicator (only log major milestones to reduce overhead)
                if frame_idx == 0 or frame_idx == state['total_frames'] - 1:
                    percent = (frame_idx + 1) / state['total_frames'] * 100
                    print(f"📹 Chunked generation: user {user_id} frame {frame_idx + 1}/{state['total_frames']} ({percent:.1f}%)")
            
            # Check if generation is complete for this user
            if state['current_frame'] >= state['total_frames']:
                state['generation_complete'] = True
                
                # Start replay mode
                cache.reset_replay()
                self.active_animations[user_id] = {
                    'type': 'replay',
                    'cache': cache,
                    'start_time': time.time(),
                    'active': True
                }
                
                # Clean up generation state
                del self.chunked_generation_state[user_id]
                
        return work_done
        
    def _check_for_stop_command_during_generation(self, user_id: int) -> bool:
        """Check if a stop command is pending for this user during frame generation
        Returns True if stop command detected, False otherwise"""
        if not self.worker_communication_dir:
            return False  # Fallback to flag-based checking if no communication dir set
            
        try:
            import json
            command_file = f"{self.worker_communication_dir}/commands.json"
            command_signal_file = f"{command_file}.signal"
            
            # Check if there's a pending command
            if os.path.exists(command_signal_file) and os.path.exists(command_file):
                with open(command_file, 'r') as f:
                    command = json.load(f)
                    
                # Check if it's a stop command for this user
                if (command.get('action') == 'stop_user_animation' and 
                    command.get('user_id') == user_id):
                    print(f"🛑 DETECTED stop command for user {user_id} during frame generation!")
                    return True
                    
        except Exception as e:
            # If file reading fails, continue with generation
            print(f"Warning: Could not check for stop commands: {e}")
            
        return False
        
    def _capture_user_frame_to_cache(self, user_id: int, frame_index: int) -> dict | None:
        """Capture a single frame for the frame cache during generation
        Returns dict of {camera_name: image_path} or None if failed"""
        if user_id not in self.user_environments:
            return None
            
        import os
        
        # Create frame-specific directory
        frame_dir = f"/tmp/isaac_worker/user_{user_id}_frames/frame_{frame_index:04d}"
        os.makedirs(frame_dir, exist_ok=True)
        
        cameras = self.user_environments[user_id]['cameras']
        captured_files = {}
        
        for camera_name, camera in cameras.items():
            try:
                rgb_data = camera.get_rgb()
                if rgb_data is not None:
                    clean_name = camera_name.lower().replace('camera_', '')
                    filename = f"{clean_name}_{frame_index:04d}.jpg"
                    filepath = os.path.join(frame_dir, filename)
                    Image.fromarray(rgb_data).save(filepath, 'JPEG', quality=85)
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
        """Reset a user's environment back to the last synchronized state
        - User 0: Resets robot + objects using stored object references  
        - Cloned users: Resets robot + objects using scene registry objects
        Both approaches ensure fresh initial state for animation restart"""
        if user_id not in self.user_environments:
            return
            
        if not hasattr(self, 'last_sync_config') or self.last_sync_config is None:
            print(f"⚠️ No sync config available for user {user_id} reset")
            return
            
        import numpy as np
        import omni.usd
        from pxr import Gf, UsdGeom
        
        try:
            env_data = self.user_environments[user_id]
            robot = env_data['robot']
            world_path = env_data['world_path']
            
            # STEP 1: Reset robot joints to synchronized state
            initial_q = np.array(self.last_sync_config.get('robot_joints', [0.0] * 7))
            initial_q = np.append(initial_q, initial_q[-1])  # Add mimic joint
            
            robot.set_joint_positions(initial_q)
            
            # STEP 2: Reset objects to fresh synchronized state for ALL environments
            # Use the correct object references for each environment type
            
            object_states = self.last_sync_config.get('object_poses', {
                "Cube_01": {"pos": [0.5, 0.0, 0.1], "rot": [0, 0, 0, 1]},
                "Cube_02": {"pos": [0.5, 0.2, 0.1], "rot": [0, 0, 0, 1]},
                "Tennis": {"pos": [0.5, -0.2, 0.1], "rot": [0, 0, 0, 1]}
            })
            
            if user_id == 0:
                # User 0: Use original object references - reset to absolute positions
                
                if 'Cube_01' in object_states and 'cube_01' in self.objects:
                    state = object_states['Cube_01']
                    pos = np.array(state["pos"])
                    rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])
                    if self.objects['cube_01'].is_valid():
                        # Reset position AND clear physics state
                        self.objects['cube_01'].set_world_pose(position=pos, orientation=rot)
                        self.objects['cube_01'].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                        self.objects['cube_01'].set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                    
                if 'Cube_02' in object_states and 'cube_02' in self.objects:
                    state = object_states['Cube_02'] 
                    pos = np.array(state["pos"])
                    rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])
                    if self.objects['cube_02'].is_valid():
                        # Reset position AND clear physics state
                        self.objects['cube_02'].set_world_pose(position=pos, orientation=rot)
                        self.objects['cube_02'].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                        self.objects['cube_02'].set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                    
                if 'Tennis' in object_states and 'tennis_ball' in self.objects:
                    state = object_states['Tennis']
                    pos = np.array(state["pos"])
                    rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])
                    if self.objects['tennis_ball'].is_valid():
                        # Reset position AND clear physics state - CRITICAL for sphere/ball objects
                        self.objects['tennis_ball'].set_world_pose(position=pos, orientation=rot)
                        self.objects['tennis_ball'].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                        self.objects['tennis_ball'].set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                
            else:
                # Cloned environments: Use scene registry objects WITH SPATIAL OFFSET
                
                # Calculate the spatial offset for this environment (same as sync logic)
                environment_spacing = 50.0
                spatial_offset = np.array([user_id * environment_spacing, user_id * environment_spacing, 0])
                
                # Reset each object using scene registry with spatial offset
                object_mappings = [
                    ("Cube_01", f"cube_01_user_{user_id}"),
                    ("Cube_02", f"cube_02_user_{user_id}"),
                    ("Tennis", f"tennis_user_{user_id}")
                ]
                
                for config_key, scene_name in object_mappings:
                    if config_key in object_states:
                        state = object_states[config_key]
                        
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
                            
            # Let physics settle after all resets are complete
            for step in range(8):
                self.world.step(render=True)
            
        except Exception as e:
            print(f"❌ Failed to reset user {user_id} environment: {e}")
        
    def update_animations(self):
        """Update all active animations (called each frame)
        NEW: Handles both legacy live animations and efficient replay system"""
        import numpy as np
        
        current_time = time.time()
        
        for user_id, anim_data in list(self.active_animations.items()):
            if not anim_data['active']:
                continue
                
            anim_type = anim_data.get('type', 'legacy')
            
            if anim_type == 'replay':
                # NEW: Frame-based replay system - no physics simulation needed
                # Frames are already generated and cached, just update timing
                cache = anim_data['cache']
                if cache.is_complete:
                    # The cache handles its own timing and looping
                    # No robot position updates needed as frames contain all visual data
                    pass
                else:
                    # If cache is not complete, something went wrong
                    print(f"⚠️ Animation cache for user {user_id} is not complete, stopping animation")
                    anim_data['active'] = False
                    
            else:
                # LEGACY: Original live physics simulation system
                elapsed = current_time - anim_data['start_time']
                progress = min(elapsed / anim_data['duration'], 1.0)
                
                # Smooth interpolation (ease-in-out)
                smooth_progress = 0.5 * (1 - np.cos(np.pi * progress))
                
                # Interpolate joint positions
                initial = anim_data['initial_joints']
                goal = anim_data['goal_joints']
                
                # Debug: Check shapes match
                if initial.shape != goal.shape:
                    print(f"⚠️ Shape mismatch in user {user_id}: initial {initial.shape} vs goal {goal.shape}")
                    continue
                    
                current_joints = initial + (goal - initial) * smooth_progress
                
                # Apply to robot
                robot = self.user_environments[user_id]['robot']
                robot.set_joint_positions(current_joints)
                
                # Loop animation when complete
                if progress >= 1.0:
                    anim_data['start_time'] = current_time  # Restart loop
                
    def capture_user_frame(self, user_id, output_dir):
        """Capture current frame for specific user
        NEW: Serves cached frames if available, falls back to live capture"""
        if user_id not in self.user_environments:
            return None
            
        import os
        
        # Check if this user has a frame cache and is in replay mode
        if (user_id in self.active_animations and 
            self.active_animations[user_id].get('type') == 'replay' and
            user_id in self.frame_caches):
            
            cache = self.frame_caches[user_id]
            current_frame_data = cache.get_current_replay_frame()
            
            if current_frame_data:
                # Serve cached frame files directly (copy to expected output location)
                os.makedirs(f"{output_dir}/user_{user_id}", exist_ok=True)
                captured_files = {}
                
                for camera_name, cached_filepath in current_frame_data.items():
                    # Copy cached file to expected output location with expected naming
                    output_filename = f"user_{user_id}_{camera_name}.jpg"
                    output_filepath = f"{output_dir}/user_{user_id}/{output_filename}"
                    
                    try:
                        # Copy the cached frame to the output location
                        import shutil
                        if os.path.exists(cached_filepath):
                            shutil.copy2(cached_filepath, output_filepath)
                            captured_files[camera_name] = output_filepath
                        else:
                            print(f"⚠️ Cached frame file not found: {cached_filepath}")
                    except Exception as e:
                        print(f"⚠️ Error copying cached frame {cached_filepath}: {e}")
                
                if captured_files:
                    # Return cached frame data
                    return captured_files
                else:
                    print(f"⚠️ Failed to serve cached frames for user {user_id}, falling back to live capture")
        
        # FALLBACK: Original live capture system (for non-cached animations or failures)
        os.makedirs(f"{output_dir}/user_{user_id}", exist_ok=True)
        
        cameras = self.user_environments[user_id]['cameras']
        captured_files = {}
        
        for camera_name, camera in cameras.items():
            rgb_data = camera.get_rgb()
            if rgb_data is not None:
                clean_name = camera_name.lower().replace('camera_', '')
                filename = f"user_{user_id}_{clean_name}.jpg"
                filepath = f"{output_dir}/user_{user_id}/{filename}"
                Image.fromarray(rgb_data).save(filepath, 'JPEG', quality=85)
                captured_files[clean_name] = filepath
                
        return captured_files
        
    def animation_loop(self, output_dir):
        """Main animation loop for streaming frames"""
        print("Starting animation loop...")
        
        frame_count = 0
        last_capture_time = time.time()
        capture_interval = 1.0 / 30.0  # 30 FPS
        
        while self.running and self.animation_mode:
            # Update physics
            self.world.step(render=True)
            
            # Update all user animations
            self.update_animations()
            
            # Capture frames at intervals
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
        """Handle runtime commands from backend"""
        action = command.get('action')
        
        if action == 'start_animation':
            return self.start_user_animation(
                user_id=command['user_id'],
                goal_joints=command.get('goal_joints'),
                duration=command.get('duration', 3.0)
            )
            
        elif action == 'stop_animation':
            return self.stop_user_animation(command['user_id'])
            
        elif action == 'terminate':
            # Clean up frame caches before terminating
            self.cleanup_all_frame_caches()
            self.running = False
            return {"status": "terminating", "caches_cleaned": True}
            
        else:
            return {"error": f"Unknown action: {action}"}