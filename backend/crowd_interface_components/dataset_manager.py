from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.control_utils import sanity_check_dataset_robot_compatibility, sanity_check_dataset_name

import datasets
from database import Database

class DatasetManager():

    def __init__(self, lerobot_cfg, dataset: LeRobotDataset = None, database: Database = None):
        self.dataset = dataset
        self.lerobot_cfg = lerobot_cfg
        self.database = database

    def init_dataset(self, cfg, robot):
        """Intialize dataset for data collection policy training"""
        if self.lerobot_cfg.resume:
            self.dataset = LeRobotDataset(
                self.lerobot_cfg.data_collection_policy_repo_id,
                root=self.lerobot_cfg.root
            )
            self.dataset.start_image_writer(
                num_processes=self.lerobot_cfg.num_image_writer_processes,
                num_threads=self.lerobot_cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )
            sanity_check_dataset_robot_compatibility(self.dataset, robot, self.lerobot_cfg.fps, self.lerobot_cfg.video)

        else:
            sanity_check_dataset_name(self.lerobot_cfg.data_collection_policy_repo_id, self.lerobot_cfg.policy)
            self.dataset = LeRobotDataset.create(
                self.lerobot_cfg.data_collection_policy_repo_id,
                self.lerobot_cfg.fps,
                root=self.lerobot_cfg.root,
                robot=robot,
                use_videos=self.lerobot_cfg.video,
                image_writer_processes=self.lerobot_cfg.num_image_writer_processes,
                image_writer_threads=self.lerobot_cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )

        # For UI fallback and dataset writes, always use self.lerobot_cfg.single_task
        self.task_text = getattr(self.lerobot_cfg, "single_task", None)

        # NEW: run the one-time context query (if enabled/configured)
        try:
            if self._vlm_enabled and self.use_vlm_prompt:
                self._run_vlm_context_once()
        except Exception as e:
            print(f"⚠️ Context VLM call failed: {e}")
        
        # Update dataset action shape to accommodate crowd responses
        self._update_dataset_action_shape()
    
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

        
    def save_episode(self, buffer):
        for state_id in sorted(buffer.keys()):
            state = buffer[state_id]
            obs = self.database._load_obs_from_disk(state['obs_path'])
            frame = {**obs, "action": state["action"], "task": state["task"]}
            self.dataset.add_frame(frame)
            self.database._delete_obs_from_disk(state.get("obs_path"))

        self.dataset.save_episode()