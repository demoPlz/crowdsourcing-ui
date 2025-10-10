from constants import *
from crowd_interface_config import CrowdInterfaceConfig

from state_manager import StateManager

class CrowdInterface():
    
    def __init__(self, cfg: CrowdInterfaceConfig):
        
        self.cfg = cfg

        self.state_manager = StateManager(cfg)