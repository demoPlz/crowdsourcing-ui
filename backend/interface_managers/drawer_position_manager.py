"""Drawer Position Manager Module.

Handles drawer position estimation using ArUco markers from observation images.
Works with saved observation data (like pose estimation) rather than direct camera access.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch


class DrawerPositionManager:
    """Manages drawer position estimation using ArUco markers.

    Estimates drawer displacement from closed position using calibrated marker positions.
    Works with observation images (like pose estimation) rather than direct camera access.

    Attributes:
        enabled: Whether drawer tracking is enabled
        drawer_joint_name: Name of prismatic joint to control (e.g., "Drawer_Joint")
        marker_positions_closed: World-frame positions of markers at closed position
        marker_size: Physical marker size in meters
        dict_name: ArUco dictionary name
        obs_camera_key: Key for the observation camera in obs_dict (e.g., "observation.images.cam_realsense")

    """

    def __init__(
        self,
        calibration_manager,
        drawer_joint_name: str = "Drawer_Joint",
        calib_file: str = "data/calib/drawer_markers_closed.json",
        repo_root: Path = None,
        obs_camera_key: str = "observation.images.cam_main",  # Key for main camera (RealSense) in obs_dict
    ):
        """Initialize drawer position manager.

        Args:
            calibration_manager: CalibrationManager instance for camera calibration
            drawer_joint_name: Name of drawer prismatic joint in simulation
            calib_file: Path to drawer calibration file (relative to repo root)
            repo_root: Repository root path
            obs_camera_key: Key for observation camera in obs_dict

        """
        self.enabled = False
        self.drawer_joint_name = drawer_joint_name
        self.calibration = calibration_manager
        self.obs_camera_key = obs_camera_key
        
        # ArUco dictionary mapping
        self.DICT_MAP = {
            "4X4_50": cv2.aruco.DICT_4X4_50,
            "4X4_100": cv2.aruco.DICT_4X4_100,
            "5X5_50": cv2.aruco.DICT_5X5_50,
            "6X6_50": cv2.aruco.DICT_6X6_50,
            "APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
        }

        # Load calibration if available
        if repo_root is None:
            repo_root = Path(__file__).resolve().parent.parent.parent
        
        calib_path = repo_root / calib_file
        
        if not calib_path.exists():
            print(f"⚠️  Drawer calibration not found: {calib_path}")
            print("    Drawer position tracking disabled. Run calibrate_drawer_markers.py to enable.")
            return
        
        try:
            with open(calib_path, 'r') as f:
                drawer_calib = json.load(f)
            
            self.marker_size = drawer_calib['marker_size_m']
            self.dict_name = drawer_calib['dict']
            
            # Extract closed positions (world frame)
            self.markers_closed = {}
            for marker_id_str, data in drawer_calib['markers'].items():
                marker_id = int(marker_id_str)
                pos_closed = np.array(data['position'], dtype=np.float64)
                self.markers_closed[marker_id] = pos_closed
            
            # Load camera intrinsics and extrinsics
            intr_file = repo_root / 'data/calib/intrinsics_realsense_d455.npz'
            extr_file = repo_root / 'data/calib/extrinsics_realsense_d455.npz'
            
            if not intr_file.exists() or not extr_file.exists():
                print(f"⚠️  Camera calibration files not found")
                print("    Drawer position tracking disabled.")
                return
            
            intr_data = np.load(intr_file)
            self.K = intr_data['K'].astype(np.float64)
            self.D = intr_data['D'].astype(np.float64)
            
            extr_data = np.load(extr_file)
            self.T_base_cam = extr_data['T_base_cam'].astype(np.float64)
            
            # Setup ArUco detection
            self.ar_dict = cv2.aruco.getPredefinedDictionary(self.DICT_MAP[self.dict_name])
            try:
                self.detector = cv2.aruco.ArucoDetector(self.ar_dict)
                self.use_new_api = True
            except (AttributeError, TypeError):
                self.use_new_api = False
            
            self.enabled = True
            print(f"✓ Drawer position tracking enabled")
            print(f"  Joint: {self.drawer_joint_name}")
            print(f"  Dictionary: {self.dict_name}")
            print(f"  Marker size: {self.marker_size}m ({self.marker_size*100}cm)")
            print(f"  Markers at closed: {list(self.markers_closed.keys())}")
            print(f"  Observation key: {self.obs_camera_key}")
            
        except Exception as e:
            print(f"⚠️  Failed to initialize drawer position tracking: {e}")
            import traceback
            traceback.print_exc()
            self.enabled = False

    def _tensor_to_rgb_uint8(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert observation tensor to RGB uint8 numpy array.

        Args:
            tensor: Observation tensor in (C, H, W) or (H, W, C) format

        Returns:
            RGB uint8 numpy array (H, W, 3)

        """
        if tensor is None:
            return None
        
        # Ensure it's a tensor
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        
        # Convert CHW to HWC if needed (like pose_worker does)
        if tensor.ndim == 3 and tensor.shape[0] == 3 and tensor.shape[-1] != 3:
            tensor = tensor.permute(1, 2, 0).contiguous()  # CHW->HWC
        
        # Convert to numpy
        img = tensor.cpu().numpy()
        
        # Normalize to [0, 255] if needed
        if img.dtype == np.float32 or img.dtype == np.float64:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        return img

    def _estimate_marker_pose(self, corners):
        """Estimate marker pose using solvePnP."""
        half_size = self.marker_size / 2.0
        obj_points = np.array([
            [-half_size,  half_size, 0],
            [ half_size,  half_size, 0],
            [ half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float64)
        
        img_points = corners.reshape(-1, 2).astype(np.float64)
        
        success, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, self.K, self.D,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        
        if success:
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()
            return T
        return None

    def estimate_from_obs(self, obs_dict: dict) -> float | None:
        """Estimate drawer position from observation dictionary.

        Args:
            obs_dict: Observation dictionary containing camera images

        Returns:
            Drawer displacement in meters (positive = open), or None if estimation fails

        """
        if not self.enabled:
            return None
        
        if not obs_dict or self.obs_camera_key not in obs_dict:
            return None
        
        try:
            # Extract and convert image
            image_tensor = obs_dict[self.obs_camera_key]
            image = self._tensor_to_rgb_uint8(image_tensor)
            
            if image is None:
                return None
            
            # Convert to grayscale for ArUco detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Detect markers
            if self.use_new_api:
                corners, ids, _ = self.detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.ar_dict)

            if ids is None or len(ids) == 0:
                return None
            
            # Process each detected marker
            drawer_distances = []
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id not in self.markers_closed:
                    continue
                
                # Estimate current pose in camera frame
                T_cam_marker = self._estimate_marker_pose(corners[i])
                if T_cam_marker is None:
                    continue
                
                # Convert to world frame
                T_world_marker_current = self.T_base_cam @ T_cam_marker
                pos_current = T_world_marker_current[:3, 3]
                
                # Get closed position
                pos_closed = self.markers_closed[marker_id]
                
                # Compute distance (full 3D distance)
                delta = pos_current - pos_closed
                distance = np.linalg.norm(delta)
                
                drawer_distances.append(distance)
            
            if not drawer_distances:
                return None
            
            # Return average distance
            avg_distance = np.mean(drawer_distances)
            print(f"🗄️  Drawer displacement estimated: {avg_distance*100:.2f} cm ({len(drawer_distances)} markers)")
            return avg_distance
            
        except Exception as e:
            print(f"⚠️  Drawer position estimation failed: {e}")
            return None

    def get_joint_position_from_obs(self, obs_dict: dict) -> dict | None:
        """Get drawer joint position from observation dictionary.

        Args:
            obs_dict: Observation dictionary containing camera images

        Returns:
            Dict with joint_name -> position, or None if estimation fails.
            Joint position is negative of displacement (drawer opens with negative values)

        """
        displacement = self.estimate_from_obs(obs_dict)
        if displacement is None:
            return None
        
        # Joint position is negative of displacement
        # e.g., 10cm displacement -> -0.1 joint position
        joint_position = -displacement
        
        print(f"🗄️  Drawer joint position: {self.drawer_joint_name} = {joint_position:.4f} m (displacement: {displacement*100:.2f} cm)")
        
        return {self.drawer_joint_name: joint_position}

