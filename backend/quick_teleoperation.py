
from __future__ import annotations

import cv2
import time
import trossen_arm
import numpy as np

from flask import Flask, jsonify
from flask_cors import CORS
from flask import request
from threading import Thread, Lock
from collections import deque
from math import cos, sin

app = Flask(__name__)
CORS(app)

states = deque(maxlen=50)

cam_ids = {
    "front":       0,   # change indices / paths as needed
    "left":        1,
    "right":       2,
    "perspective": 3,
}
cams = {}

latest_goal = None
goal_lock = Lock()

def make_views() -> dict[str, list]:
    """Four NxMx3 pink RGB images (here 64 x 64 x 3)."""
    H, W = 64, 64
    pink = np.array([255, 105, 180], dtype=np.uint8)  # hot-pink [R,G,B]
    img  = np.broadcast_to(pink, (H, W, 3)).copy()    # (64,64,3)
    # convert to nested lists so Flask’s JSON encoder can handle it
    return {name: img.tolist()
            for name in ("left", "right", "front", "perspective")}

def init_cameras():
    """Open all cameras once; skip any that fail."""
    for name, idx in cam_ids.items():
        cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
        if cap.isOpened():
            cams[name] = cap
        else:
            print(f"⚠️  camera “{name}” (id {idx}) could not be opened")

def grab_frame(cap, size=(64, 64)) -> np.ndarray | None:
    ok, frame = cap.read()
    if not ok:
        return None
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)   # WxH
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def get_views() -> dict[str, list]:
    """Return a 64×64 RGB image dict from available webcams."""
    H, W = 64, 64
    pink = np.broadcast_to([255, 105, 180], (H, W, 3)).astype(np.uint8)

    views = {}
    for name in ("left", "right", "front", "perspective"):
        if name in cams:
            frame = grab_frame(cams[name])
            views[name] = (frame if frame is not None else pink).tolist()
        else:
            views[name] = pink.tolist()
    return views

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


def make_camera_poses() -> dict[str, list]:
    return {
        #           x     y     z     roll   pitch   yaw
        "front_pose":       euler_pose(1.0, 0.0, 0.15, 0.0, -np.pi/2 - 0.1, -np.pi/2),
        "left_pose":        euler_pose(0.2, -1.0, 0.15, -np.pi/2, 0.0, 0.0),
        "right_pose":       euler_pose(0.2,  1.0, 0.15, np.pi/2, 0.0, np.pi),
        "perspective_pose": euler_pose(1.3,  1.0, 1.0, np.pi/4, -np.pi/4, -3*np.pi/4),
    }


def make_controls() -> list[str]:
    # any subset of ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    return ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']

@app.route("/api/get-state")
def get_state():
    if not states:
        return jsonify({})
    return jsonify(states.popleft())


@app.route("/api/submit-goal", methods=["POST"])
def submit_goal():
    '''
    {
        "joint_positions": { "joint_0": 0.12, … },
        "gripper": -1,        # -1 close, +1 open, 0 unchanged
        "ee_pose": {
            "position":   [x, y, z],
            "quaternion": [qx, qy, qz, qw]
        }
    }
    '''
    
    data = request.get_json(force=True, silent=True) or {}
    print("🔔 Confirmed goal received:", data, flush=True)

    global latest_goal
    with goal_lock:
        latest_goal = data

    return jsonify({"status": "ok"})

def robot_loop():
    global latest_goal

    server_ip = '192.168.1.3'

    print("Initializing the drivers...")
    driver = trossen_arm.TrossenArmDriver()

    print("Configuring the drivers...")
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_leader,
        server_ip,
        False
    )

    convert_rad = np.pi / 180.0

    print("Moving to home positions...")
    driver.set_all_modes(trossen_arm.Mode.position)
    driver.set_all_positions(
        np.array([0.0, 60.0, 75.0, -60.0, 0.0, 0.0, 2.0]) * convert_rad,
        2.0,
        True
    )

    position = driver.get_cartesian_positions()

    driver.set_arm_modes(trossen_arm.Mode.position)
    driver.set_gripper_mode(trossen_arm.Mode.position)

    driver.set_gripper_position(0.044, 0.0, False) # default to open
    gripper_motion = 1 # open

    print("Opening webcams...")
    init_cameras()


    print("Starting to teleoperate the robots...")

    try:
        dead_zone   = 0             # ignore tiny noise around center

        last_t = time.time()
        while True:
            vec = list(driver.get_all_positions())          # VectorDouble → Python list
            joint_names = [                                 # must match URDF / front-end
                "joint_0", "joint_1", "joint_2",
                "joint_3", "joint_4", "joint_5",
                "left_carriage_joint"
            ]
            joint_map = { n: v for n, v in zip(joint_names, vec) }
            
            states.append({
                "joint_positions": joint_map,
                "views":           get_views(),
                "camera_poses":    make_camera_poses(),
                "gripper":         gripper_motion,
                'controls':        make_controls()
            })

            with goal_lock:
                submission = latest_goal
                latest_goal = None

            if submission:
                joint_names = [
                    "joint_0", "joint_1", "joint_2",
                    "joint_3", "joint_4", "joint_5",
                    "left_carriage_joint"
                ]
                goal_positions = np.array(
                    [submission["joint_positions"][n] for n in joint_names],
                    dtype=float
                )

                driver.set_all_positions(
                    goal_positions,
                    goal_time=2.0,
                    blocking=False
                )

                # gripper
                if submission.get("gripper") in (-1, +1):
                    target = 0.044 if submission["gripper"] > 0 else 0.000
                    driver.set_gripper_position(target, 0.0, False)
                    gripper_motion = submission["gripper"]

                
    except KeyboardInterrupt:
        pass
    
    finally:
        for cap in cams.values():
            cap.release()
    
    print('Resetting...')
    print("Moving to home positions...")
    driver.set_all_modes(trossen_arm.Mode.position)
    driver.set_all_positions(
        np.array([0.0, np.pi/2, np.pi/2, 0.0, 0.0, 0.0, 0.0]),
        2.0,
        True
    )

    print("Moving to sleep positions...")
    driver.set_all_modes(trossen_arm.Mode.position)
    driver.set_all_positions(
        np.zeros(driver.get_num_joints()),
        2.0,
        True
    )


if __name__=='__main__':

    # --------------------------------------------------------------------------- #
    #  Run Flask server in background so main loop keeps control of the robot    #
    # --------------------------------------------------------------------------- #

    server_thread = Thread(
        target=lambda: app.run(host="0.0.0.0", port=9000, debug=False, use_reloader=False),
        daemon=True
    )
    server_thread.start()
    robot_loop()