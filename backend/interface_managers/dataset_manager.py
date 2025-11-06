"""
Dataset Manager Module

Manages LeRobot dataset operations for the crowd interface.
Handles dataset initialization, episode saving, and observation loading/cleanup.
"""

import os
import torch
import datasets
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.control_utils import sanity_check_dataset_robot_compatibility, sanity_check_dataset_name


class DatasetManager:
    """
    Manages LeRobot dataset operations.
    
    Responsibilities:
    - Dataset initialization (create or resume)
    - Episode saving (frames and episodes)
    - Observation loading from disk cache
    - Observation cleanup after saving
    - Dataset action shape updates for crowd responses
    
    Attributes:
        dataset: LeRobotDataset instance
        task_text: Task text used for dataset frames
        required_responses_per_critical_state: Number of responses per critical state (for action shape)
        obs_cache_root: Root directory for observation cache
    """
    
    def __init__(
        self,
        required_responses_per_critical_state: int,
        obs_cache_root: Path,
    ):
        """
        Initialize dataset manager.
        
        Args:
            required_responses_per_critical_state: Number of responses per critical state (for action shape)
            obs_cache_root: Root directory for observation cache
        """
        self.required_responses_per_critical_state = required_responses_per_critical_state
        self._obs_cache_root = obs_cache_root
        
        # Dataset state
        self.dataset = None
        self.task_text = None
    
    # =========================
    # Dataset Initialization
    # =========================
    
    def init_dataset(self, cfg, robot):
        """Initialize dataset for data collection policy training"""
        if cfg.resume:
            self.dataset = LeRobotDataset(
                cfg.data_collection_policy_repo_id,
                root=cfg.root
            )
            self.dataset.start_image_writer(
                num_processes=cfg.num_image_writer_processes,
                num_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )
            sanity_check_dataset_robot_compatibility(self.dataset, robot, cfg.fps, cfg.video)

        else:
            sanity_check_dataset_name(cfg.data_collection_policy_repo_id, cfg.policy)
            self.dataset = LeRobotDataset.create(
                cfg.data_collection_policy_repo_id,
                cfg.fps,
                root=cfg.root,
                robot=robot,
                use_videos=cfg.video,
                image_writer_processes=cfg.num_image_writer_processes,
                image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )

        # For UI fallback and dataset writes, always use cfg.single_task
        self.task_text = getattr(cfg, "single_task", None)
        
        # Update dataset action shape to accommodate crowd responses
        self._update_dataset_action_shape()
        
        return self.task_text
    
    def _update_dataset_action_shape(self):
        """Update the dataset's action feature shape to include crowd responses dimension"""
        if self.dataset is not None and "action" in self.dataset.features:
            from datasets import Sequence, Value, Features
            from lerobot.common.datasets.utils import get_hf_features_from_features
            
            original_action_dim = self.dataset.features["action"]["shape"][-1]  # Get the last dimension (joint count)
            new_action_shape = (self.required_responses_per_critical_state * original_action_dim,)
            
            # Update both the dataset features and metadata
            self.dataset.features["action"]["shape"] = new_action_shape
            self.dataset.meta.features["action"]["shape"] = new_action_shape
            
            # Recreate the HF dataset with updated features
            if self.dataset.hf_dataset is not None:
                # Get new HF features from the updated self.features
                new_hf_features = get_hf_features_from_features(self.dataset.features)
                
                # Create a new empty dataset with the correct features
                ft_dict = {col: [] for col in new_hf_features}
                new_hf_dataset = datasets.Dataset.from_dict(ft_dict, features=new_hf_features, split="train")
                
                # Apply the same transform
                from lerobot.common.datasets.utils import hf_transform_to_torch
                new_hf_dataset.set_transform(hf_transform_to_torch)
                
                # Replace the old dataset
                self.dataset.hf_dataset = new_hf_dataset

            print(f"📐 Updated dataset action shape to {new_action_shape} (crowd_responses={self.required_responses_per_critical_state}, joints={original_action_dim})")
    
    # =========================
    # Episode Saving
    # =========================
    
    def save_episode(self, buffer):
        """Save episode from completed states buffer to dataset"""
        for state_id in sorted(buffer.keys()):
            state = buffer[state_id]
            obs = self.load_obs_from_disk(state['obs_path'])
            if 'depth' in obs:
                del obs['depth']  # delete the depth tensor
            frame = {**obs, "action": state["action_to_save"], "task": state["task_text"]}
            self.dataset.add_frame(frame)
            self._delete_obs_from_disk(state.get("obs_path"))

        self.dataset.save_episode()
    
    # =========================
    # Observation Cache Management
    # =========================
    
    def load_obs_from_disk(self, path: str | None) -> dict:
        """Load observations from disk cache"""
        if not path:
            return {}
        try:
            return torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"⚠️  failed to load obs from {path}: {e}")
            return {}

    def _delete_obs_from_disk(self, path: str | None):
        """Delete observation file from disk cache after saving"""
        if not path:
            return
        try:
            os.remove(path)
        except Exception:
            pass
