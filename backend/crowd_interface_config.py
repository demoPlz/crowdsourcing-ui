import argparse
import sys
from pathlib import Path


class CrowdInterfaceConfig():

    def __init__(self):
        # Task
        self.task_name = "drawer"  # one word
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

        # Sim
        self.use_sim: bool = True

        # Objects. Can only be configured in __init__ and not overriden
        self.objects: dict[str, str] = {
            "Cube_Blue": "Blue cube", 
            "Cube_Red": "Red cube", 
            "Tennis": "Tennis ball"
        } # {name of xform: name for langsam}
        
        # Resolve mesh paths to absolute paths
        repo_root = Path(__file__).resolve().parent.parent
        self.object_mesh_paths: dict[str, str] = {
            "Cube_Blue": str((repo_root / "public/assets/cube.obj").resolve()),
            "Cube_Red": str((repo_root / "public/assets/cube.obj").resolve()),
            "Tennis":  str((repo_root / "public/assets/sphere_new.obj").resolve())
        }

    @classmethod
    def from_cli_args(cls, argv=None):
        """
        Create a CrowdInterfaceConfig instance with CLI overrides.
        Parses only the crowd-specific CLI args and removes them from sys.argv.
        """
        # Parse CLI args that can override config
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--task-name", type=str, help="One-word task name")
        parser.add_argument("--required-responses-per-critical-state", type=int)
        parser.add_argument("--autofill-critical-states", action="store_true")
        parser.add_argument("--num-autofill-actions", type=int)
        parser.add_argument("--use-manual-prompt", action="store_true")
        parser.add_argument("--show-demo-videos", action="store_true")
        parser.add_argument("--record-ui-demo-videos", action="store_true")
        parser.add_argument("--ui-demo-videos-dir", type=str)
        parser.add_argument("--clear-ui-demo-videos-dir", action="store_true")
        parser.add_argument("--use_sim", action="store_true")
        
        args, remaining = parser.parse_known_args(argv if argv is not None else sys.argv[1:])
        
        # Remove our args from sys.argv so LeRobot doesn't see them
        sys.argv = [sys.argv[0]] + remaining
        
        # Create config instance
        config = cls()
        
        # Apply CLI overrides
        if args.task_name is not None:
            config.task_name = args.task_name
        if args.required_responses_per_critical_state is not None:
            config.required_responses_per_critical_state = args.required_responses_per_critical_state
        if args.autofill_critical_states:
            config.autofill_critical_states = True
        if args.num_autofill_actions is not None:
            config.num_autofill_actions = args.num_autofill_actions
        if args.use_manual_prompt:
            config.use_manual_prompt = True
        if args.show_demo_videos:
            config.show_demo_videos = True
        if args.record_ui_demo_videos:
            config.record_ui_demo_videos = True
        if args.ui_demo_videos_dir is not None:
            config.ui_demo_videos_dir = args.ui_demo_videos_dir
        if args.clear_ui_demo_videos_dir:
            config.clear_ui_demo_videos_dir = True
        if args.use_sim:
            config.use_sim = True
            
        return config

    def to_crowd_interface_kwargs(self) -> dict:
        """
        Convert this config to kwargs suitable for CrowdInterface constructor.
        Maps config fields to the expected CrowdInterface parameter names.
        """
        kwargs = {
            "task_name": self.task_name,
            "required_responses_per_state": self.required_responses_per_state,
            "required_responses_per_critical_state": self.required_responses_per_critical_state,
            "autofill_critical_states": self.autofill_critical_states,
            "num_autofill_actions": self.num_autofill_actions,
            "use_manual_prompt": self.use_manual_prompt,
            "show_demo_videos": self.show_demo_videos,
            "use_sim": self.use_sim,
            "objects": self.objects,
            "object_mesh_paths": self.object_mesh_paths
        }
        
        # Handle demo video recording settings
        if self.record_ui_demo_videos:
            kwargs["record_demo_videos"] = True
            if self.ui_demo_videos_dir is not None:
                kwargs["demo_videos_dir"] = self.ui_demo_videos_dir
            if self.clear_ui_demo_videos_dir:
                kwargs["demo_videos_clear"] = True
        
        return kwargs