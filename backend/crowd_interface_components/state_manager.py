from crowd_interface import CrowdInterface
from crowd_interface_config import CrowdInterfaceConfig
from camera_manager import CameraManager
from database import Database
from monitoring_manager import MonitoringManager
from dataset_manager import DatasetManager

from constants import *

from threading import Thread, Lock, Timer
import queue
import torch
import time
import random

class StateManager():
    def __init__(self,
                 crowd_interface: CrowdInterface,
                 camera_manager: CameraManager,
                 database: Database,
                 dataset_manager: DatasetManager):
        
        # Save overall cfg
        self.interface_cfg = crowd_interface.cfg

        # Width of the dataset
        self.required_responses_per_state = self.interface_cfg.required_responses_per_state
        self.required_responses_per_critical_state = self.interface_cfg.required_responses_per_critical_state
        
        # Autofill: the act of labeling a state by duplicating its existing labels
        self.autofill_critical_states = self.interface_cfg.autofill_critical_states
        if self.interface_cfg.autofill_critical_states:
            self.num_autofill_actions = self.interface_cfg.num_autofill_actions

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

        # Handling saving episode data to Lerobot dataset after completion
        self._episode_finalize_timers: dict[int, Timer] = {}


        self._start_auto_label_worker()

        self.crowd_interface = crowd_interface
        self.camera_manager = camera_manager
        self.dataset_manager = dataset_manager
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
        self.camera_manager.push_obs_view("obs_main",  obs_dict.get("observation.images.cam_main"))
        self.camera_manager.push_obs_view("obs_wrist", obs_dict.get("observation.images.cam_wrist"))
        
        state_info = {
            # Identity
            "state_id": state_id,
            "episode_id": episode_id,

            # Robot state
            "joint_positions": joint_positions_float,
            "gripper": gripper_motion,
            "controls": ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper'], # legacy, will remove
            "left_carriage_external_force": left_carriage_external_force,

            # Observations
            "obs_path": obs_path,

            # Views
            "view_paths": view_paths,

            # Labels
            "actions": [],
            "responses_received": 0,

            # Critical state fields
            "critical": False,
            "prompt_ready": False,
            "text_prompt": None, # replaces flex_text_prompt
            "video_prompt": None, # replaces flex_video_id

            # Task
            "task": self.task_text

            # No other fields; segmentation, and all others, no longer supported\
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

            if episode_id not in self.pending_states_by_episode or \
            state_id not in self.pending_states_by_episode[episode_id]:
                # State already fully labeled
                return

            state_info = self.pending_states_by_episode[episode_id][state_id]

            required_responses = self.required_responses_per_critical_state if state_info['critical'] else self.required_responses_per_state
            
            joint_positions = response_data['joint_positions']
            gripper_action = response_data['gripper']

            state_info["responses_received"] += 1

            goal_positions = []
            for joint_name in JOINT_NAMES:
                joint_value = joint_positions[joint_name]
                goal_positions.append(float(joint_value[0]))

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
                if state_info['critical'] and state_id == self.next_state_id - 1:
                    # Choose action to execute (a_execute) at random
                    # Shift chose action to the front of the array
                    a_execute_index = random.randint(0, required_responses - 1)
                    state_info["actions"][0], state_info["actions"][a_execute_index] = state_info["actions"][a_execute_index], state_info["actions"][0]
                    self.interface.latest_goal = state_info["actions"][:required_responses][0]

                all_actions = torch.cat(state_info["actions"][:required_responses], dim=0)

                if required_responses < self.required_responses_per_critical_state:
                    # Pad unimportant states's action tensor
                    missing_responses = self.required_responses_per_critical_state - required_responses
                    action_dim = len(JOINT_NAMES)
                    padding_size = missing_responses * action_dim
                    padding = torch.full((padding_size,), float('nan'), dtype=torch.float32)
                    all_actions = torch.cat([all_actions, padding], dim=0)

                state_info['action_to_save'] = all_actions

                # Save to completed states buffer (for forming training set)
                if episode_id not in self.completed_states_buffer_by_episode:
                    self.completed_states_buffer_by_episode[episode_id] = {}
                self.completed_states_buffer_by_episode[episode_id][state_id] = state_info

                # Save to completed states (for monitoring)
                if episode_id not in self.completed_states_by_episode:
                    self.completed_states_by_episode[episode_id] = {}
                self.completed_states_by_episode[episode_id][state_id] = state_info

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

            self.demote_earlier_unanswered_criticals(latest_state_id, latest_episode_id)
            self.auto_label_previous_states(latest_state_id)

    def demote_earlier_unanswered_criticals(self, current_state_id, episode_id):
        '''
        Demote critical states before state_id in episode with episode_id to non-critical
        '''
        for state_id in self.pending_states_by_episode[episode_id].keys():
            if state_id < current_state_id \
                  and self.pending_states_by_episode[episode_id][state_id]['critical'] \
                  and not self.pending_states_by_episode[episode_id][state_id]['actions']:
                self.pending_states_by_episode[episode_id][state_id]['critical'] = False

    def _schedule_episode_finalize_after_grace(self, episode_id: int):
        delay = self.episode_finalize_grace_s
        timer = Timer(delay, self._finalize_episode_if_still_empty, args=(episode_id,))
        timer.daemon = True
        self._episode_finalize_timers[episode_id] = timer
        timer.start()
    
    def _cancel_episode_finalize_timer(self, episode_id: int):
        t = self._episode_finalize_timers.pop(episode_id, None)
        if t:
            t.cancel()

    def _finalize_episode_if_still_empty(self, episode_id: int):
        """
        Timer callback
        """
        with self.state_lock:
            self._episode_finalize_timers.pop(episode_id, None)

            if self.pending_states_by_episode.get(episode_id):
                # New states has become pending in the episode
                return
            
            self.episodes_completed.add(episode_id) # for monitoring

            buffer = self.completed_states_buffer_by_episode[episode_id]
            self.dataset_manager.save_episode(buffer)

            del self.completed_states_buffer_by_episode[episode_id]

    def get_latest_state(self) -> dict:
        """
        Get a pending state from current serving episode
        We only implement crowd mode, meaning that we serve the last state
        of the last episode always.
        """

        with self.state_lock:
            episode_id = self.current_serving_episode
            state_id = self.next_state_id - 1

            if episode_id not in self.pending_states_by_episode \
            or state_id not in self.pending_states_by_episode[episode_id]:
                # No pending states left
                return {
                    "status": "no_pending_states",
                    "blocked_critical_states": False
                }
            
            state_info = self.pending_states_by_episode[episode_id][state_id]

            if state_info['critical'] and not state_info['prompt_ready']:
                # There are pending states but no ready states

                return {
                    "status": "no_ready_states",
                    "blocked_critical_states": True,
                }
            
            # Return the latest state for labeling
            return state_info.copy()
            
            
    """
    Auto-labeling thread
    """
    def auto_label_previous_states(self, critical_state_id):
        self.auto_label_queue.put_nowait(critical_state_id)

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
            episode_id = max(self.pending_states_by_episode.keys())

            episode_states = self.pending_states_by_episode[episode_id]

            template_action = None

            previous_critical_id_in_episode = []
            for state_id in episode_states.keys():
                if episode_states[state_id]['critical'] \
                    and state_id < critical_state_id \
                    and len(episode_states[state_id]['actions']) > 0:
                    previous_critical_id_in_episode.append(state_id)

            if previous_critical_id_in_episode: # Previous critical states exist
                latest_critical_state = episode_states[max(previous_critical_id_in_episode)]
                template_action = latest_critical_state['actions'][0]
            else: # This is the first critical state in the episode
                first_state_id = min(episode_states.keys())
                first_state = episode_states[first_state_id]
                joint_positions = first_state['state']['joint_positions']
                gripper_action = first_state['state']['gripper']
                goal_positions = []
                for joint_name in JOINT_NAMES:
                    joint_value = joint_positions[joint_name]
                    goal_positions.append(float(joint_value))

                goal_positions[-1] = 0.044 if gripper_action > 0 else 0.0
                template_action = torch.tensor(goal_positions, dtype=torch.float32)

            states_to_label = []
            for state_id, state_info in episode_states.items():
                if state_id < critical_state_id and not state_info['critical']:
                    states_to_label.append(state_id)

            for state_id in states_to_label:
                state_info = episode_states[state_id]

                while state_info["responses_received"] < self.required_responses_per_state:
                    state_info["actions"].append(template_action.clone())
                    state_info['responses_received'] += 1

                all_actions = torch.cat(state_info["actions"][:self.required_responses_per_state], dim=0)
                        
                # Pad with inf values to match critical state shape
                missing_responses = self.required_responses_per_critical_state - self.required_responses_per_state
                action_dim = len(JOINT_NAMES)
                padding_size = missing_responses * action_dim
                padding = torch.full((padding_size,), float('nan'), dtype=torch.float32)
                all_actions = torch.cat([all_actions, padding], dim=0)
                
                completed_state = {
                    "obs_path": state_info.get("obs_path"),
                    "action": all_actions,
                    "task": self.task_text if self.task_text else "crowdsourced_task",
                }
                
                if episode_id not in self.completed_states_buffer_by_episode:
                    self.completed_states_buffer_by_episode[episode_id] = {}
                self.completed_states_buffer_by_episode[episode_id][state_id] = completed_state
                
                completed_state_monitor = {
                    "responses_received": state_info["responses_received"],
                    "is_critical": state_info['critical'],
                }

                # Move to episode-based completed states
                if episode_id not in self.completed_states_by_episode:
                    self.completed_states_by_episode[episode_id] = {}
                self.completed_states_by_episode[episode_id][state_id] = completed_state_monitor

                del self.pending_states_by_episode[episode_id][state_id]
            
