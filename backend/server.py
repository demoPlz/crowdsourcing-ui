# ---------- server.py  -------------------------------------------------------
from __future__ import annotations
import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS
import random
from math import cos, sin

app = Flask(__name__)
CORS(app)

# --------------------------------------------------------------------------- #
#  Helpers to fabricate the state                                             #
# --------------------------------------------------------------------------- #
def make_joint_positions() -> dict[str, float]:
    """Simulated joint angles (rad). Keys must match URDF joint names."""
    deg = np.deg2rad  # convenience
    return {
        "joint_0": deg(   0.0),
        "joint_1": deg(  61.0),
        "joint_2": deg(  73.0),
        "joint_3": deg(  61.0),
        "joint_4": deg(   0.0),
        "joint_5": deg(   0.0),
        "left_carriage_joint": 0.020, # 0 = fully closed, 0.044 m = fully open
    }


def make_views() -> dict[str, list]:
    """Four N×M×3 pink RGB images (here 64 × 64 × 3)."""
    H, W = 64, 64
    pink = np.array([255, 105, 180], dtype=np.uint8)  # hot-pink [R,G,B]
    img  = np.broadcast_to(pink, (H, W, 3)).copy()    # (64,64,3)
    # convert to nested lists so Flask’s JSON encoder can handle it
    return {name: img.tolist()
            for name in ("left", "right", "front", "perspective")}

def euler_pose(x: float, y: float, z: float,
               roll: float, pitch: float, yaw: float) -> list[list[float]]:
    """
    Build a 4×4 **world** matrix (row-major list-of-lists) from
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
        "front_pose":       euler_pose(0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        # "left_pose":        euler_pose(0.0, -2.0, 1.0, 0.0, 0.0, np.pi/2),
        # "right_pose":       euler_pose(0.0,  2.0, 1.0, 0.0, 0.0,-np.pi/2),
        # "perspective_pose": euler_pose(2.0,  2.0, 2.0, -0.4, -0.3, 0.8),
    }


def choose_axis() -> str:
    """Pick one of the control axes at random (placeholder)."""
    return random.choice(["x", "y", "z", "roll", "pitch", "yaw", "gripper"])


# --------------------------------------------------------------------------- #
#  API endpoint                                                               #
# --------------------------------------------------------------------------- #
@app.route("/api/get-state")
def get_state():
    state = {
        "joint_positions": make_joint_positions(),
        "views":           make_views(),
        "camera_poses":    make_camera_poses(),
        "axis":            choose_axis(),
    }
    return jsonify(state)


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9000)
