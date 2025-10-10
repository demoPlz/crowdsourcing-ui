from crowd_interface import CrowdInterface
from crowd_interface_config import CrowdInterfaceConfig
from camera_manager import CameraManager
from database import Database
from monitoring_manager import MonitoringManager

from constants import *

from threading import Thread, Lock
import queue
import torch
import time
import random

class StateManager():
    def __init__(self,
                 crowd_interface: CrowdInterface,
                 camera_manager: CameraManager,
                 database: Database,
                 monitoring_manager: MonitoringManager):
        
        # Save overall cfg
        self.interface_cfg = crowd_interface.cfg

        # Width of the dataset
        self.required_responses_per_state = self.cfg.required_responses_per_state
        self.required_responses_per_critical_state = self.cfg.required_responses_per_critical_state
        
        # Autofill: the act of labeling a state by duplicating its existing labels
        self.autofill_critical_states = self.cfg.autofill_critical_states
        if self.cfg.autofill_critical_states:
            self.num_autofill_actions = self.cfg.num_autofill_actions

        # Episode-based state management
        self.pending_states_by_episode: dict[dict] = {}
        self.completed_states_by_episode: dict[dict] = {}
        self.completed_states_buffer_by_episode: dict[dict] = {}

        # Global state id tracking
        self.next_state_id = 0

        # Lock for safe multithreading
        self.state_lock = Lock()

        # Track episodes for monitor
        self.episodes_completed = set()
        self.episodes_being_completed = set()

        # Auto-labeling: the act of labeling a state with another state's existing label
        self.auto_label_queue = queue.Queue()
        self.auto_label_worker_thread = None
        self.auto_label_worker_running = False


        self._start_auto_label_worker()

        self.crowd_interface = crowd_interface
        self.camera_manager = camera_manager
        self.monitoring_manager = monitoring_manager
        self.database = database
    
    """
    Add State
    """
    def add_state(self,
                  joint_positions: dict,
                  gripper_motion: int = None,
                  obs_dict: dict[str, torch.Tensor] = None,
                  episode_id: str = None,
                  left_carriage_external_force: float | None = None):
        '''
        Called by lerobot code to add states to backend.
        '''
        joint_positions_float = {k: float(v) for k, v in joint_positions.items()}

        frontend_state = {
            "joint_positions": joint_positions_float,
            "gripper": gripper_motion,
            "controls": ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper'], # legacy
            "left_carriage_external_force": left_carriage_external_force
        }

        state_id = self.next_state_id
        self.next_state_id += 1

        # Persist views to disk to avoid storing in memory
        view_paths = self.database.persist_views_to_disk(episode_id, state_id, self.camera_manager._snapshot_latest_views()) # legacy
        
        # Persist obs to disk
        obs_dict_deep_copy = {}
        for key, value in obs_dict.items():
            obs_dict_deep_copy[key] = value.clone().detach()
        obs_path = self.database.persist_obs_to_disk(episode_id, state_id, obs_dict_deep_copy)
        del obs_dict_deep_copy

        # Push obs to monitoring frontend
        self.monitoring_manager.push_obs_view("obs_main",  obs_dict.get("observation.images.cam_main"))
        self.monitoring_manager.push_obs_view("obs_wrist", obs_dict.get("observation.images.cam_wrist"))
        
        state_info = {
            "state": frontend_state.copy(),
            "obs_path": obs_path,
            "view_paths": view_paths,
            "actions": [],
            "responses_received": 0,
            "critical": False,
            "episode_id": episode_id
        }

        with self.state_lock:
            # Initialize episode containers if needed
            if episode_id not in self.pending_states_by_episode:
                self.pending_states_by_episode[episode_id] = {}
                self.completed_states_by_episode[episode_id] = {}
                
            # Add state to pending states
            self.pending_states_by_episode[episode_id][state_id] = state_info

            self.current_serving_episode = episode_id

    def record_response(self, response_data: dict):
        '''
        Record a response for a specific state. Handles all the side-effects.
        '''

        with self.state_lock:
            state_id = response_data['state_id']
            episode_id = response_data['episode_id']

            state_info = self.pending_states_by_episode[episode_id][state_id]

            if not state_info:
                # State already fully labeled
                return False

            required_responses = self.required_responses_per_critical_state if state_info['critical'] else self.required_responses_per_state
            
            joint_positions = response_data['joint_positions']
            gripper_action = response_data['gripper']

            state_info["responses_received"] += 1

            goal_positions = []
            for joint_name in JOINT_NAMES:
                joint_value = joint_positions[joint_name]
                goal_positions.append(float(joint_value))

            goal_positions[-1] = 0.044 if gripper_action > 0 else 0.0
            goal_positions = torch.tensor(goal_positions, dtype=torch.float32)
            state_info['actions'].append(goal_positions)

            # Autofill
            if state_info["critical"] and self.autofill_critical_states:
                remaining = state_info["responses_received"]
                clones_to_add = min(self.num_autofill_actions - 1, remaining)
                for _ in range(clones_to_add):
                    state_info["actions"].append(goal_positions.clone())
                state_info['responses_received'] += clones_to_add
                
            # Handle completion
            if state_info['responses_received'] >= required_responses:
                all_actions = torch.cat(state_info["actions"][:required_responses], dim=0)

                if state_info['critical'] and state_id == self.next_state_id - 1:
                    # Choose action to execute (a_execute) at random
                    
                    self.interface.latest_goal = random.choice(state_info["actions"][:required_responses])

                if required_responses < self.required_responses_per_critical_state:
                    # Pad unimportant states's action tensor
                    missing_responses = self.required_responses_per_critical_state - required_responses
                    action_dim = len(JOINT_NAMES)
                    padding_size = missing_responses * action_dim
                    padding = torch.full((padding_size,), float('nan'), dtype=torch.float32)
                    all_actions = torch.cat([all_actions, padding], dim=0)

                # Everything we need to later write states to Lerobot dataset in order
                completed_state = {
                    "obs_path": state_info.get("obs_path"),
                    "action": all_actions,
                    "task": self.interface_cfg.task_text # Text label for frame
                }

                # Save to completed states buffer (for forming training set)
                if episode_id not in self.completed_states_buffer_by_episode:
                    self.completed_states_buffer_by_episode[episode_id] = {}
                self.completed_states_buffer_by_episode[episode_id][state_id] = completed_state

                completed_state_monitor = { # Contains legacy components
                    "responses_received": state_info["responses_received"],
                    "completion_time": time.time(),
                    "is_critical": state_info['crtical'],
                    "prompt_ready": True,
                    "flex_text_prompt": state_info['flex_text_prompt'] if 'flex_text_prompt' in state_info else None,
                    "flex_video_id": state_info['flex_video_id'] if 'flex_video_id' in state_info else None,
                }
                # Save to completed states (for monitoring)
                if episode_id not in self.completed_states_by_episode:
                    self.completed_states_by_episode[episode_id][state_id] = completed_state_monitor

                
                # Remove from pending
                del self.pending_states_by_episode[episode_id][state_id]

                # Handle episode completion
                if not self.pending_states_by_episode[episode_id]:
                    self._schedule_episode_finalize_after_grace(episode_id)
                
            

    def set_last_state_to_critical(self):
        with self.state_lock:
            
            if self.pending_states_by_episode:
                latest_episode_id = max(self.pending_states_by_episode.keys())
                episode_states = self.pending_states_by_episode[latest_episode_id]
                if episode_states:
                    latest_state_id = max(episode_states.keys())
            else:
                # self.pending_states_by_episode hasn't been populated yet
                return
            
            info = self.pending_states_by_episode[latest_episode_id][latest_state_id]
            if info['critical']:
                # Already set
                return
            info['critical'] = True

            self.demote_earlier_unanswered_criticals()
            self.auto_label_previous_states(latest_state_id)

    def _schedule_episode_finalize_after_grace(self, episode_id):
        pass
            

    def auto_label_previous_states(self, critical_state_id):
        self.auto_label_queue.put_nowait(critical_state_id)

    """
    Auto-labeling thread
    """
    def _start_auto_label_worker(self):
        self.auto_label_worker_thread = Thread(target=self._auto_label_worker, daemon=True)
        self.auto_label_worker_thread.start()

    def _auto_label_worker(self):
        for critical_state_id in iter(self.auto_label_queue.get, None):
            self._auto_label(critical_state_id)

    def _auto_label(self, critical_state_id):
        '''
        Given critical_state_id, auto-labels noncritical states in the same episode before the critical state with:"
        1. The executed action of the previous important state
        2. If no previous important state exists, the joint positions of the first state in the episode
        '''
        with self.state_lock:
            episode_id = self.next_state_id - 1

            episode_states = self.pending_states_by_episode[episode_id]

            for i in range(critical_state_id - 1, -1, -1):
                if episode_states[i]['critical']:
                    break

            if episode_states[i]['action']