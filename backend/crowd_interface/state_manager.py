from crowd_interface_config import CrowdInterfaceConfig
from camera_manager import CameraManager
from database import Database
from monitoring_manager import MonitoringManager

from threading import Thread, Lock
import queue
import torch

class StateManager():
    def __init__(self, 
                 cfg: CrowdInterfaceConfig, 
                 camera_manager: CameraManager,
                 database: Database,
                 monitoring_manager: MonitoringManager):

        # Width of the dataset
        self.required_responses_per_state = cfg.required_responses_per_state
        self.required_responses_per_critical_state = cfg.required_responses_per_critical_state
        
        # Autofill: the act of labeling a state by duplicating its existing labels
        self.autofill_critical_states = cfg.autofill_critical_states
        if cfg.autofill_critical_states:
            self.num_autofill_actions = cfg.num_autofill_actions

        # Episode-based state management
        self.pending_states_by_episode = {}
        self.completed_states_by_episode = {}
        self.completed_states_buffer_by_episode = {}

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
            "observations": None,
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

    
