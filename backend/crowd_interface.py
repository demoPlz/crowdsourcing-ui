from __future__ import annotations


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

class CrowdInterface():
    '''
    Sits between the frontend and the backend
    '''
    def __init__(self):

        self.states = deque()
        self.cams = {}
        self.latest_goal = None
        self.goal_lock = Lock()
        self._gripper_motion = 1  # Initialize gripper motion

        self._camera_poses = self._make_camera_poses()


        # Precompute immutable views and camera poses to avoid per-tick allocations
        H, W = 64, 64
        pink = np.full((H, W, 3), 255, dtype=np.uint8)  # one-time NumPy buffer
        blank = pink.tolist()                           # one-time JSON-serializable view
        # Reuse the same object for all cameras (read-only downstream)
        self._blank_views = {
            "left":        blank,
            "right":       blank,
            "front":       blank,
            "perspective": blank,
        }


    ### ---Camera Management---

    def init_cameras(self):
        """Open all cameras once; skip any that fail."""
        for name, idx in CAM_IDS.items():
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            if cap.isOpened():
                self.cams[name] = cap
            #     print(f"✓ Camera '{name}' opened successfully")
            # else:
            #     print(f"⚠️  camera “{name}” (id {idx}) could not be opened")

    def cleanup_cameras(self):
        """Close all cameras"""
        for cap in self.cams.values():
            cap.release()
        self.cams.clear()
    
    def get_views(self) -> dict[str, list]:
        """Return a 64x64 RGB image dict from available webcams."""
        H, W = 64, 64
        pink = np.broadcast_to([255, 255, 255], (H, W, 3)).astype(np.uint8)

        views = {}
        for name in ("left", "right", "front", "perspective"):
            # if name in cams:
            #     frame = _grab_frame(cams[name])
            #     views[name] = (frame if frame is not None else pink).tolist()
            # else:
                # views[name] = pink.tolist()

            views[name] = pink.tolist()
        return views
    
    def _grab_frame(cap, size=(64, 64)) -> np.ndarray | None:
        ok, frame = cap.read()
        if not ok:
            return None
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)   # WxH
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    # --- State Management ---
    def add_state(self, joint_positions: dict, gripper_motion: int = None):
        if gripper_motion is not None:
            self._gripper_motion = int(gripper_motion)

        # Cheap, explicit cast of the 7 scalars to built-in floats
        jp = {k: float(v) for k, v in joint_positions.items()}

        self.states.append({
            "joint_positions": jp,
            "views": self._blank_views,       # reuse precomputed JSON-serializable views
            "camera_poses": self._camera_poses,  # reuse precomputed poses
            "gripper": self._gripper_motion,
            "controls": ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper'],
        })
        # print(f"🟢 State added at {current_time}, total states: {len(self.states)}")
        # print(f"🟢 Joint positions: {joint_positions}")
        # print(f"🟢 Gripper: {self._gripper_motion}")
    
    def get_latest_state(self) -> dict:
        """Get the latest state (pops from queue)"""
        # print(f"🔍 get_latest_state called - states length: {len(self.states)}")
        if not self.states:
            # print("🔍 No states available, returning empty dict")
            return {}
        latest = self.states[-1]
        # print(f"🔍 Returning latest state with keys: {list(latest.keys())}")
        return latest
    
    # --- Goal Management ---
    def submit_goal(self, goal_data: dict):
        """Submit a new goal from the frontend"""
        self.latest_goal = goal_data
        # print(f"🔔 Goal received: {goal_data}")
    
    def get_latest_goal(self) -> dict | None:
        """Get and clear the latest goal (for robot loop to consume)"""
        goal = self.latest_goal
        self.latest_goal = None
        return goal
    
    def has_pending_goal(self) -> bool:
        """Check if there's a pending goal"""
        return self.latest_goal is not None
    
    # --- Helper Methods ---

    def _make_camera_poses(self) -> dict[str, list]: #TODO Placeholders for now
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
        
        return {
            #           x     y     z     roll   pitch   yaw
            "front_pose":       euler_pose(1.0, 0.0, 0.15, 0.0, -np.pi/2 - 0.1, -np.pi/2),
            "left_pose":        euler_pose(0.2, -1.0, 0.15, -np.pi/2, 0.0, 0.0),
            "right_pose":       euler_pose(0.2,  1.0, 0.15, np.pi/2, 0.0, np.pi),
            "perspective_pose": euler_pose(1.3,  1.0, 1.0, np.pi/4, -np.pi/4, -3*np.pi/4),
        }
    

    
def create_flask_app(crowd_interface: CrowdInterface) -> Flask:
    """Create and configure Flask app with the crowd interface"""
    app = Flask(__name__)
    CORS(app)
    
    @app.route("/api/get-state")
    def get_state():
        import time
        current_time = time.time()
        state = crowd_interface.get_latest_state()
        # print(f"🔍 Flask route /api/get-state called at {current_time}")
        # print(f"🔍 crowd_interface.states length: {len(crowd_interface.states)}")
        # if len(crowd_interface.states) > 0:
        #     print(f"🔍 Latest state joint_positions: {crowd_interface.states[-1].get('joint_positions', 'NO_JOINTS')}")
        #     print(f"🔍 Latest state gripper_action: {crowd_interface.states[-1]['gripper']}")
        response = jsonify(state)
        # Prevent caching
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    
    @app.route("/api/test")
    def test():
        return jsonify({"message": "Flask server is working", "states_count": len(crowd_interface.states)})
    
    @app.route("/api/submit-goal", methods=["POST"])
    def submit_goal():
        data = request.get_json(force=True, silent=True) or {}
        crowd_interface.submit_goal(data)
        return jsonify({"status": "ok"})
    
    return app