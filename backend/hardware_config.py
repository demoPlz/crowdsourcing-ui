"""
Hardware Configuration Constants

Camera IDs and calibration file paths for the physical setup.
Modify these values to match your specific hardware configuration.
"""

# Camera device indices (change to match your system's USB camera enumeration)
CAM_IDS = {
    "front":       18,
    "left":        4,
    "right":       2,
    "perspective": 1,
}

# Real hardware calibration file paths
# Each camera needs intrinsics (for undistortion) and extrinsics (for 3D positioning)
REAL_CALIB_PATHS = {
    "front": {
        "intr": "data/calib/intrinsics_front_1_640x480.npz",
        "extr": "data/calib/extrinsics_front_1.npz",
    },
    "left": {
        "intr": "data/calib/intrinsics_left_2_640x480.npz",
        "extr": "data/calib/extrinsics_left_2.npz",
    },
    "right": {
        "intr": "data/calib/intrinsics_right_3_640x480.npz",
        "extr": "data/calib/extrinsics_right_3.npz",
    },
    "perspective": {
        "intr": "data/calib/intrinsics_perspective_4_640x480.npz",
        "extr": "data/calib/extrinsics_perspective_4.npz",
    },
}

# Simulation calibration file paths (Isaac Sim virtual cameras)
SIM_CALIB_PATHS = {
    "front": "data/calib/calibration_front_sim.json",
    "left": "data/calib/calibration_left_sim.json",
    "right": "data/calib/calibration_right_sim.json",
    "top": "data/calib/calibration_top_sim.json",
}
