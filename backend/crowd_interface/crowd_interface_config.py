from constants import *

class CrowdInterfaceConfig():

    def __init__(self):

        # Task
        self.task_name = "drawer" # one word
        self.task_text = "Put the objects on the desk into the middle drawer."

        # Number of responses per state settings
        self.required_responses_per_state: int = 1
        self.required_responses_per_critical_state: int = 10

        # Critical state autofill settings
        self.autofill_critical_states: bool = False
        self.num_autofill_actions: int | None = None

        # UI Prompt control settings
        self.use_manual_prompt: bool = False
        self.show_demo_videos: bool = False

        # UI setting
        self.simulated_frontend: bool = True

        # UI usage video demonstration recording settings
        self.record_ui_demo_videos: bool = False
        self.ui_demo_videos_dir: str | None = None
        self.clear_ui_demo_videos_dir: bool = False