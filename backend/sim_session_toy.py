# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from PIL import Image
from isaacsim import SimulationApp

# Start the simulation
simulation_app = SimulationApp({"headless": False})

import requests
import numpy as np
import carb
import omni.usd
from omni.isaac.core import World
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.articulations import Articulation
from pxr import Gf, UsdGeom, Usd

# Import the Camera class
from isaacsim.sensors.camera import Camera, get_all_camera_objects

# --- Configuration ---
USD_PATH = "/home/yilong/crowdsourcing-ui/public/assets/usd/drawer.usd"
BACKEND_URL = "http://127.0.0.1:9000/initial-state"

ROBOT_PATH = "/World/wxai"
OBJ_CUBE_01_PATH = "/World/Cube_01"
OBJ_CUBE_02_PATH = "/World/Cube_02"
OBJ_TENNIS_PATH = "/World/Tennis"

# --- Helper Functions (omitted for brevity, no changes needed) ---
def get_initial_state_from_backend():
    """Fetches the initial state from your backend service."""
    try:
        print(f"Fetching initial state from {BACKEND_URL}")
        response = requests.get(BACKEND_URL, timeout=5)
        response.raise_for_status()
        state = response.json()
        print("Successfully fetched initial state from backend")
        return state
    except requests.exceptions.RequestException as e:
        carb.log_error(f"Failed to fetch initial state from backend: {str(e)}. Using default state.")
        return {
            "q": [0.0] * 7,  # Default 7-DOF joint positions
            "objects": {
                "Cube_01": {"pos": [0.5, 0.0, 0.1], "rot": [0, 0, 0, 1]},
                "Cube_02": {"pos": [0.5, 0.2, 0.1], "rot": [0, 0, 0, 1]},
                "Tennis": {"pos": [0.5, -0.2, 0.1], "rot": [0, 0, 0, 1]}
            }
        }
    
def hide_robot():
    """Alternative: Move robot far away (preserves everything)"""
    hide_robot.original_pos, hide_robot.original_rot = robot.get_world_pose()
    hide_robot.current_joint_positions = robot.get_joint_positions()
    # Move robot below ground
    robot.set_world_pose(position=np.array([0, 0, -100]), orientation=hide_robot.original_rot)

def show_robot():
    """Restore robot to original position"""
    robot.set_world_pose(position=hide_robot.original_pos, 
                        orientation=hide_robot.original_rot)
    robot.set_joint_positions(hide_robot.current_joint_positions)
        
# --- Main Simulation Logic ---

# Load the USD stage FIRST
print(f"Loading environment from {USD_PATH}")
omni.usd.get_context().open_stage(USD_PATH)

# Wait for the stage to load
for i in range(20):
    simulation_app.update()

# Create the World object AFTER the stage is loaded
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
print("Stage loaded and World object created.")

# Fetch the initial state
initial_state = get_initial_state_from_backend()
initial_q = np.array(initial_state["q"])
initial_q = np.append(initial_q, initial_q[-1])
object_states = initial_state["objects"]

# Get handles to the prims
robot = world.scene.add(Articulation(prim_path=ROBOT_PATH, name="widowx_robot"))
cube_01 = world.scene.add(RigidPrim(prim_path=OBJ_CUBE_01_PATH, name="cube_01"))
cube_02 = world.scene.add(RigidPrim(prim_path=OBJ_CUBE_02_PATH, name="cube_02"))
tennis_ball = world.scene.add(RigidPrim(prim_path=OBJ_TENNIS_PATH, name="tennis_ball"))

cameras = {}
stage = omni.usd.get_context().get_stage()
all_cameras = get_all_camera_objects(root_prim='/')
cameras = {all_cameras[i].name: all_cameras[i] for i in range(len(all_cameras))}

# Reset world and set initial poses
world.reset()

# Initialize cameras AFTER world.reset()
for camera in all_cameras:
    camera.initialize()
    camera.set_resolution((1080,1080))
    original_h_aperture = camera.get_horizontal_aperture()
    original_v_aperture = camera.get_vertical_aperture()
    
    # Apply zoom by reducing aperture size
    zoom_factor = 1 / 2.5  # 1.7 x zoom
    camera.set_horizontal_aperture(original_h_aperture * zoom_factor)
    camera.set_vertical_aperture(original_v_aperture * zoom_factor)
    
    camera.add_rgb_to_frame()

robot.set_joint_positions(initial_q)

# Set object poses
if cube_01.is_valid():
    state = object_states.get("Cube_01")
    if state:
        pos = np.array(state["pos"])
        rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]]) # WXYZ
        cube_01.set_world_pose(position=pos, orientation=rot)

if cube_02.is_valid():
    state = object_states.get("Cube_02")
    if state:
        pos = np.array(state["pos"])
        rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]]) # WXYZ
        cube_02.set_world_pose(position=pos, orientation=rot)
        
if tennis_ball.is_valid():
    state = object_states.get("Tennis")
    if state:
        pos = np.array(state["pos"])
        rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]]) # WXYZ
        tennis_ball.set_world_pose(position=pos, orientation=rot)

print("\nScene setup complete. Starting simulation loop...")
# --- Simulation Loop ---

hide_robot()
for step in range(10):
    world.step(render=True)

# Capture initial images (robot hidden)
initial_front_rgb = cameras['Camera_Front'].get_rgb()
initial_left_rgb = cameras['Camera_Left'].get_rgb()
initial_right_rgb = cameras['Camera_Right'].get_rgb()
initial_top_rgb = cameras['Camera_Top'].get_rgb()

# Save initial images
Image.fromarray(initial_front_rgb).save('initial_front_image.png')
Image.fromarray(initial_left_rgb).save('initial_left_image.png')
Image.fromarray(initial_right_rgb).save('initial_right_image.png')
Image.fromarray(initial_top_rgb).save('initial_top_image.png')

show_robot()
for step in range(10):
    world.step(render=True)

# Capture final images (robot visible)
front_rgb = cameras['Camera_Front'].get_rgb()
left_rgb = cameras['Camera_Left'].get_rgb()
right_rgb = cameras['Camera_Right'].get_rgb()
top_rgb = cameras['Camera_Top'].get_rgb()

# Save final images
Image.fromarray(front_rgb).save('front_image.png')
Image.fromarray(left_rgb).save('left_image.png')
Image.fromarray(right_rgb).save('right_image.png')
Image.fromarray(top_rgb).save('top_image.png')

while simulation_app.is_running():
    # This step advances physics and renders the scene
    world.step(render=True)

simulation_app.close()