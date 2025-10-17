#!/usr/bin/env python3
"""
Isaac Sim worker script with two modes:
1. Static image capture (for frontend IK interaction)
2. Animation mode (for physics simulation with cloning)
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
        self.ik_solver = None
        # State management for reuse
        self.simulation_initialized = False
        self.objects = {}  # Store object references for reuse
        self.hide_robot_funcs = None  # Store hide/show functions
        self.simulation_app = simulation_app  # Store simulation app reference
        
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
        """Capture images with current state (robot hidden)"""
        import os
        from PIL import Image
        
        if not self.simulation_initialized:
            raise RuntimeError("Must call initialize_simulation() first")
            
        os.makedirs(output_dir, exist_ok=True)
        
        print("Capturing images with current state...")
        
        # Hide robot (frontend will render it)
        self.hide_robot_funcs['hide']()
        
        # Let physics settle with robot hidden
        for step in range(10):
            self.world.step(render=True)

        # Capture static images
        front_rgb = self.cameras['Camera_Front'].get_rgb()
        left_rgb = self.cameras['Camera_Left'].get_rgb()
        right_rgb = self.cameras['Camera_Right'].get_rgb()
        top_rgb = self.cameras['Camera_Top'].get_rgb()

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
        """Mode 1: Capture static images (robot hidden) for frontend IK
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
        from omni.isaac.cloner import Cloner
        from omni.isaac.motion_generation import RmpFlow
        import numpy as np
        
        print(f"Initializing animation mode with {max_users} user environments...")
        
        # Show robot for animation mode - need to restore if it was hidden
        # For now, assume robot is already visible from static capture mode
        for step in range(5):
            self.world.step(render=True)
        
        # Initialize IK solver (RmpFlow for collision-aware IK)
        self.ik_solver = RmpFlow(
            robot_description_path=self.robot.robot_prim.GetPrimPath(),
            urdf_path=None,  # Isaac Sim will use robot description
            rmpflow_config_path=None,  # Use default config
            end_effector_frame_name="ee_link"  # Adjust based on your robot
        )
        self.ik_solver.initialize()
        
        # Initialize cloner
        cloner = Cloner()
        
        # Clone environments for each user
        source_prim_path = "/World"
        for user_id in range(max_users):
            if user_id == 0:
                # User 0 uses original environment
                target_path = "/World"
            else:
                # Clone for other users
                target_path = f"/World_User_{user_id}"
                cloner.clone(
                    source_prim_path=source_prim_path,
                    prim_paths=[target_path],
                    copy_from_source=True
                )
            
            # Get robot for this environment
            robot_path = f"{target_path}/wxai"
            user_robot = self.world.scene.add(Articulation(
                prim_path=robot_path, 
                name=f"robot_user_{user_id}"
            ))
            
            # Get cameras for this environment  
            user_cameras = {}
            from isaacsim.sensors.camera import get_all_camera_objects
            all_user_cameras = get_all_camera_objects(root_prim=target_path)
            for camera in all_user_cameras:
                camera.initialize()
                camera.set_resolution((640, 480))
                camera.add_rgb_to_frame()
                user_cameras[camera.name] = camera
            
            # Store environment data
            self.user_environments[user_id] = {
                'robot': user_robot,
                'cameras': user_cameras,
                'world_path': target_path
            }
            
        self.animation_mode = True
        print(f"Animation mode initialized with {max_users} environments")
        
    def start_user_animation(self, user_id, goal_pose=None, goal_joints=None, duration=3.0):
        """Start animation for specific user using IK or direct joint control"""
        if user_id not in self.user_environments:
            return {"error": f"User {user_id} environment not found"}
            
        import numpy as np
        
        robot = self.user_environments[user_id]['robot']
        current_joints = robot.get_joint_positions()
        
        # Use IK to compute goal joints from pose, or use provided joints
        if goal_pose is not None:
            # Use Isaac Sim's IK solver
            target_position = np.array(goal_pose['position'])
            target_orientation = np.array(goal_pose['orientation'])  # quaternion [x,y,z,w]
            
            # Compute IK solution
            ik_result = self.ik_solver.compute_inverse_kinematics(
                target_position=target_position,
                target_orientation=target_orientation,
                current_joint_positions=current_joints
            )
            
            if ik_result.success:
                goal_joints = ik_result.joint_positions
                print(f"IK solved for user {user_id}: {goal_joints}")
            else:
                return {"error": f"IK failed for user {user_id}"}
                
        elif goal_joints is None:
            return {"error": "Must provide either goal_pose or goal_joints"}
            
        # Store animation parameters
        self.active_animations[user_id] = {
            'initial_joints': current_joints.copy(),
            'goal_joints': np.array(goal_joints),
            'duration': duration,
            'start_time': time.time(),
            'active': True
        }
        
        return {"status": "animation_started", "user_id": user_id}
        
    def stop_user_animation(self, user_id):
        """Stop animation for specific user"""
        if user_id in self.active_animations:
            self.active_animations[user_id]['active'] = False
            del self.active_animations[user_id]
        return {"status": "animation_stopped", "user_id": user_id}
        
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
                goal_pose=command.get('goal_pose'),
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='JSON config file path')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for images')
    parser.add_argument('--mode', type=str, choices=['static', 'animation', 'both'], default='both',
                       help='Operation mode: static images only, animation only, or both')
    parser.add_argument('--max-users', type=int, default=8, help='Maximum users for animation mode')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Start simulation
    global simulation_app
    simulation_app = SimulationApp({"headless": True})
    
    try:
        worker = IsaacSimWorker()
        
        if args.mode in ['static', 'both']:
            # Capture static images first
            print("=== STATIC IMAGE CAPTURE MODE ===")
            static_result = worker.capture_static_images(config, args.output_dir)
            print(json.dumps(static_result))  # For backend to capture
            
        if args.mode in ['animation', 'both']:
            # Initialize animation mode (can reuse same environment)
            print("=== ANIMATION MODE ===")
            worker.initialize_animation_mode(max_users=args.max_users)
            
            # Signal ready for animation (backend can start accepting commands)
            ready_signal = {"status": "animation_ready", "max_users": args.max_users}
            print(json.dumps(ready_signal))
            
            # Enter animation loop (handles commands and streams frames)
            worker.animation_loop(args.output_dir)
            
        print("Worker completed successfully")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if 'simulation_app' in globals():
            simulation_app.close()

if __name__ == "__main__":
    main()