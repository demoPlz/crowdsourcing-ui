#!/usr/bin/env python3
"""
Isaac Sim worker script with two modes:
1. Static image capture (for frontend interaction)
2. Animation mode (for physics simulation with direct joint control)
"""

import sys
import json
import argparse
import time
import threading
import signal
from collections import defaultdict
from PIL import Image
from isaacsim import SimulationApp

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
        
    def start_user_animation(self, user_id, goal_joints, duration=3.0):
        """Start animation for specific user using direct joint control
        ALWAYS starts fresh from the synchronized initial state"""
        if user_id not in self.user_environments:
            return {"error": f"User {user_id} environment not found"}
            
        import numpy as np
        
        robot = self.user_environments[user_id]['robot']
        
        if goal_joints is None:
            return {"error": "Must provide goal_joints"}
        
        # Convert goal_joints to numpy array
        goal_joints = np.array(goal_joints)
        
        # STEP 1: Stop any existing animation for this user to ensure clean start
        if user_id in self.active_animations:
            print(f"🔄 Stopping existing animation for user {user_id} to start fresh")
            self.active_animations[user_id]['active'] = False
            del self.active_animations[user_id]
        
        # STEP 2: CRITICAL - Reset ENTIRE environment (robot + objects) to fresh synchronized state
        # This ensures every animation starts from the exact same scene state
        print(f"🆕 PERFORMING FULL FRESH RESET for user {user_id} before animation")
        self._reset_user_environment_to_sync_state(user_id)
        
        # STEP 3: Get the fresh initial state from the latest synchronized state
        # This ensures every animation starts from the current frontend state
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
        # Frontend sends 7 values, but robot has 8 (last joint is mimic)
        if len(goal_joints) == 7:
            # Duplicate the last joint value for the mimic joint
            goal_joints = np.append(goal_joints, goal_joints[-1])
            print(f"Extended goal_joints from 7 to 8 dimensions: {goal_joints}")
        elif len(goal_joints) != 8:
            return {"error": f"Goal joints dimension mismatch: got {len(goal_joints)}, expected 7 or 8"}
            
        print(f"🆕 FRESH START - Initial: {initial_joints}, Goal: {goal_joints}")
        
        # STEP 4: Additional verification - robot position should already be set by reset but confirm
        robot.set_joint_positions(initial_joints)
        
        # Let physics settle to ensure clean initial state
        for step in range(3):
            self.world.step(render=True)
            
        print(f"✅ Full environment {user_id} reset to fresh initial state: robot + objects")
        
        # STEP 4: Store animation parameters for fresh animation
        self.active_animations[user_id] = {
            'initial_joints': initial_joints.copy(),
            'goal_joints': goal_joints,
            'duration': duration,
            'start_time': time.time(),
            'active': True
        }
        
        return {"status": "animation_started", "user_id": user_id, "fresh_start": True}
        
    def stop_user_animation(self, user_id):
        """Stop animation for specific user and reset to fresh synchronized state"""
        if user_id in self.active_animations:
            self.active_animations[user_id]['active'] = False
            del self.active_animations[user_id]
            print(f"🛑 Stopped animation for user {user_id}")
            
            # Reset this user's environment back to the fresh synchronized state
            self._reset_user_environment_to_sync_state(user_id)
            print(f"🔄 Reset user {user_id} to fresh synchronized state")
            
        return {"status": "animation_stopped", "user_id": user_id, "reset_to_fresh": True}
        
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
            
            print(f"🔄 FULL FRESH RESET for user {user_id} - robot AND objects")
            
            # STEP 1: Reset robot joints to synchronized state
            initial_q = np.array(self.last_sync_config.get('robot_joints', [0.0] * 7))
            initial_q = np.append(initial_q, initial_q[-1])  # Add mimic joint
            
            print(f"🔄 RESETTING user {user_id} robot to fresh state: {initial_q}")
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
                print(f"🔧 Resetting objects for user 0 (original environment) using stored object references")
                
                if 'Cube_01' in object_states and 'cube_01' in self.objects:
                    state = object_states['Cube_01']
                    pos = np.array(state["pos"])
                    rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])
                    if self.objects['cube_01'].is_valid():
                        # Reset position AND clear physics state
                        self.objects['cube_01'].set_world_pose(position=pos, orientation=rot)
                        self.objects['cube_01'].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                        self.objects['cube_01'].set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                        print(f"✅ User 0 Cube_01 reset to fresh pos: {pos} (physics cleared)")
                    
                if 'Cube_02' in object_states and 'cube_02' in self.objects:
                    state = object_states['Cube_02'] 
                    pos = np.array(state["pos"])
                    rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])
                    if self.objects['cube_02'].is_valid():
                        # Reset position AND clear physics state
                        self.objects['cube_02'].set_world_pose(position=pos, orientation=rot)
                        self.objects['cube_02'].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                        self.objects['cube_02'].set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                        print(f"✅ User 0 Cube_02 reset to fresh pos: {pos} (physics cleared)")
                    
                if 'Tennis' in object_states and 'tennis_ball' in self.objects:
                    state = object_states['Tennis']
                    pos = np.array(state["pos"])
                    rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]])
                    if self.objects['tennis_ball'].is_valid():
                        # Reset position AND clear physics state - CRITICAL for sphere/ball objects
                        self.objects['tennis_ball'].set_world_pose(position=pos, orientation=rot)
                        self.objects['tennis_ball'].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                        self.objects['tennis_ball'].set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                        print(f"✅ User 0 Tennis ball reset to fresh pos: {pos} (physics cleared)")
                
            else:
                # Cloned environments: Use scene registry objects WITH SPATIAL OFFSET
                print(f"🔧 Resetting objects for user {user_id} (cloned environment) using scene registry WITH OFFSET")
                
                # Calculate the spatial offset for this environment (same as sync logic)
                environment_spacing = 50.0
                spatial_offset = np.array([user_id * environment_spacing, user_id * environment_spacing, 0])
                print(f"📍 User {user_id} reset spatial offset: {spatial_offset}")
                
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
                        
                        print(f"🔄 Reset {config_key}: Original pos {original_pos} + offset {spatial_offset} = {offset_pos}")
                        
                        # Try to get object from scene registry
                        if self.world.scene.object_exists(scene_name):
                            scene_obj = self.world.scene.get_object(scene_name)
                            # Reset position AND clear physics state
                            scene_obj.set_world_pose(position=offset_pos, orientation=rot)
                            scene_obj.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                            scene_obj.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                            print(f"✅ User {user_id} {config_key} reset via scene registry to offset pos: {offset_pos} (physics cleared)")
                        else:
                            print(f"⚠️ Scene object {scene_name} not found for user {user_id} - object may not be registered")
                            
                print(f"📍 User {user_id} objects reset to fresh synchronized state WITH SPATIAL OFFSET")
            
            # Let physics settle after all resets are complete
            for step in range(8):
                self.world.step(render=True)
            
            print(f"✅ FULL FRESH RESET COMPLETE for user {user_id} - robot AND objects at synchronized state")
            
        except Exception as e:
            print(f"❌ Failed to reset user {user_id} environment: {e}")
        
    def update_animations(self):
        """Update all active animations (called each frame)"""
        import numpy as np
        
        current_time = time.time()
        
        for user_id, anim_data in list(self.active_animations.items()):
            if not anim_data['active']:
                continue
                
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
        """Capture current frame for specific user"""
        if user_id not in self.user_environments:
            return None
            
        import os
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
            self.running = False
            return {"status": "terminating"}
            
        else:
            return {"error": f"Unknown action: {action}"}