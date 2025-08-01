# ---------- server.py  -------------------------------------------------------
from __future__ import annotations
import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS
import random

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
    }


def make_views() -> dict[str, list]:
    """Four N×M×3 pink RGB images (here 64 × 64 × 3)."""
    H, W = 64, 64
    pink = np.array([255, 105, 180], dtype=np.uint8)  # hot-pink [R,G,B]
    img  = np.broadcast_to(pink, (H, W, 3)).copy()    # (64,64,3)
    # convert to nested lists so Flask’s JSON encoder can handle it
    return {name: img.tolist()
            for name in ("left", "right", "front", "perspective")}


def random_se3() -> list[list[float]]:
    """Return a random 4×4 homogeneous transform as nested Python lists."""
    # random rotation via axis-angle
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    theta = random.uniform(-np.pi, np.pi)
    K = np.array([[    0, -axis[2],  axis[1]],
                  [ axis[2],      0, -axis[0]],
                  [-axis[1], axis[0],     0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

    # random translation in a 1 m cube
    t = np.random.uniform(-0.5, 0.5, 3)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T.tolist()


def make_camera_poses() -> dict[str, list]:
    return {name + "_pose": random_se3()
            for name in ("left", "right", "front", "perspective")}


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
