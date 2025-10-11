from constants import *
from backend.crowd_interface_config import CrowdInterfaceConfig

from state_manager import StateManager
from camera_manager import CameraManager
from database import Database

class CrowdInterface():
    
    def __init__(self, cfg: CrowdInterfaceConfig):
        
        self.cfg = cfg

        self.database = Database(cfg)

        self.latest_goal = None # For action execution
    
    def get_latest_goal(self):
        """Get and clear the latest goal (for robot loop to consume)"""
        goal = self.latest_goal
        self.latest_goal = None
        return goal
    
    def has_pending_goal(self) -> bool:
        """Check if there's a pending goal"""
        return self.latest_goal is not None