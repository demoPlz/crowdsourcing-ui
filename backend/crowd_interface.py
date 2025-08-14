CAM_IDS = {
    "front":       0,   # change indices / paths as needed
    "left":        1,
    "right":       2,
    "perspective": 3,
}

JOINT_NAMES = [
    "joint_0", "joint_1", "joint_2",
    "joint_3", "joint_4", "joint_5",
    "left_carriage_joint"
]


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

import argparse

### HELPERS

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

class CrowdInterface():
    '''
    Sits between the frontend and the backend
    '''
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)

        self.states = deque()

        self.cams = {}

        self.latest_goal = None
        self.goal_lock = Lock()


    ### ---CAMERAS---

    def init_cameras(self):
        """Open all cameras once; skip any that fail."""
        for name, idx in CAM_IDS.items():
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            if cap.isOpened():
                self.cams[name] = cap
            else:
                print(f"⚠️  camera “{name}” (id {idx}) could not be opened")

    def grab_frame(cap, size=(64, 64)) -> np.ndarray | None:
        ok, frame = cap.read()
        if not ok:
            return None
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)   # WxH
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    def get_views(self) -> dict[str, list]:
        """Return a 64x64 RGB image dict from available webcams."""
        H, W = 64, 64
        pink = np.broadcast_to([255, 255, 255], (H, W, 3)).astype(np.uint8)

        views = {}
        for name in ("left", "right", "front", "perspective"):
            # if name in cams:
            #     frame = grab_frame(cams[name])
            #     views[name] = (frame if frame is not None else pink).tolist()
            # else:
                # views[name] = pink.tolist()

            views[name] = pink.tolist()
        return views

    def make_camera_poses(self) -> dict[str, list]: #TODO Placeholders for now
        return {
            #           x     y     z     roll   pitch   yaw
            "front_pose":       euler_pose(1.0, 0.0, 0.15, 0.0, -np.pi/2 - 0.1, -np.pi/2),
            "left_pose":        euler_pose(0.2, -1.0, 0.15, -np.pi/2, 0.0, 0.0),
            "right_pose":       euler_pose(0.2,  1.0, 0.15, np.pi/2, 0.0, np.pi),
            "perspective_pose": euler_pose(1.3,  1.0, 1.0, np.pi/4, -np.pi/4, -3*np.pi/4),
        }
    
    @app.route("/api/get-state")
    def get_state(self):
        if not states:
            return jsonify({})
        return jsonify(states.popleft())