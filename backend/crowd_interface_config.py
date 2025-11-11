"""Configuration for CrowdInterface crowd-sourced data collection."""

import argparse
import sys
from pathlib import Path


class CrowdInterfaceConfig:
    """Configuration for crowd-sourced robot data collection interface.

    This config manages settings for:
    - Task definition and labeling requirements
    - Critical state annotation (user responses per state)
    - Demo video recording and display
    - Simulation integration
    - Object tracking and pose estimation

    """

    def __init__(self):
        # ========== Task Settings ==========
        self.task_name: str = "drawer"  # Single-word identifier for the task
        self.task_text: str = "Put the objects on the desk into the middle drawer."

        # ========== Labeling Requirements ==========
        self.required_responses_per_state: int = 1  # Non-critical states
        self.required_responses_per_critical_state: int = 10  # Critical states requiring multiple labels

        # ========== Critical State Autofill ==========
        # When enabled, critical states receive num_autofill_actions + 1 responses (cloned) per response
        self.autofill_critical_states: bool = False
        self.num_autofill_actions: int | None = None

        # ========== UI Prompting ==========
        self.use_manual_prompt: bool = False  # Manual text/video prompt selection per state
        self.show_demo_videos: bool = False  # Display reference videos to users

        # ========== Demo Video Recording ==========
        # Records user interaction videos for training/demonstration purposes
        self.record_ui_demo_videos: bool = False
        self.ui_demo_videos_dir: str | None = None  # Defaults to data/prompts/{task_name}/videos
        self.clear_ui_demo_videos_dir: bool = False  # Clear directory on startup

        # ========== Simulation ==========
        self.use_sim: bool = True  # Use Isaac Sim for state simulaion

        # ========== Object Tracking ==========
        # Object names and their language descriptions for pose estimation
        # Note: Keys must match USD prim names in Isaac Sim
        self.objects: dict[str, str] = {"Cube_Blue": "Blue cube", "Cube_Red": "Red cube", "Tennis": "Tennis ball"}

        # Resolve mesh paths for pose estimation (relative to repo root)
        repo_root = Path(__file__).resolve().parent.parent
        self.object_mesh_paths: dict[str, str] = {
            "Cube_Blue": str((repo_root / "public" / "assets" / "cube.obj").resolve()),
            "Cube_Red": str((repo_root / "public" / "assets" / "cube.obj").resolve()),
            "Tennis": str((repo_root / "public" / "assets" / "sphere.obj").resolve()),
        }

        # ========== Joint Tracking ==========
        # Track prismatic joint positions of drawer for drawer task
        self.joint_tracking: list = ["Drawer_Joint"]

    @classmethod
    def from_cli_args(cls, argv=None):
        """Create a CrowdInterfaceConfig instance with CLI overrides.

        Parses crowd-specific CLI arguments and removes them from sys.argv,
        allowing LeRobot's argument parser to process remaining args without conflicts.

        Args:
            argv: Command-line arguments to parse (defaults to sys.argv[1:])

        Returns:
            CrowdInterfaceConfig: Configuration instance with CLI overrides applied

        """
        parser = argparse.ArgumentParser(add_help=False)

        # Task settings
        parser.add_argument("--task-name", type=str, help="Single-word task identifier (e.g., 'drawer', 'pick_place')")

        # Labeling settings
        parser.add_argument(
            "--required-responses-per-critical-state",
            type=int,
            help="Number of user responses required for critical states",
        )
        parser.add_argument(
            "--autofill-critical-states",
            action="store_true",
            help="Auto-complete critical states after partial responses",
        )
        parser.add_argument(
            "--num-autofill-actions", type=int, help="Number of responses before auto-fill (default: all required)"
        )

        # UI settings
        parser.add_argument(
            "--use-manual-prompt", action="store_true", help="Require manual text/video prompt selection per state"
        )
        parser.add_argument("--show-demo-videos", action="store_true", help="Display reference videos to labelers")

        # Demo video recording
        parser.add_argument("--record-ui-demo-videos", action="store_true", help="Record user interaction videos")
        parser.add_argument(
            "--ui-demo-videos-dir", type=str, help="Directory for demo videos (default: data/prompts/{task}/videos)"
        )
        parser.add_argument(
            "--clear-ui-demo-videos-dir", action="store_true", help="Clear demo videos directory on startup"
        )

        # Simulation
        parser.add_argument("--use-sim", action="store_true", help="Enable Isaac Sim integration")

        args, remaining = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

        # Remove crowd-specific args from sys.argv for downstream parsers
        sys.argv = [sys.argv[0]] + remaining

        # Create config and apply overrides
        config = cls()

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
        """Convert configuration to CrowdInterface constructor kwargs.

        Returns:
            dict: Keyword arguments for CrowdInterface.__init__()

        """
        kwargs = {
            # Core settings
            "task_name": self.task_name,
            "required_responses_per_state": self.required_responses_per_state,
            "required_responses_per_critical_state": self.required_responses_per_critical_state,
            # Autofill
            "autofill_critical_states": self.autofill_critical_states,
            "num_autofill_actions": self.num_autofill_actions,
            # UI
            "use_manual_prompt": self.use_manual_prompt,
            "show_demo_videos": self.show_demo_videos,
            # Simulation
            "use_sim": self.use_sim,
            # Object tracking
            "objects": self.objects,
            "object_mesh_paths": self.object_mesh_paths,
        }

        # Optional: demo video recording (only include if enabled)
        if self.record_ui_demo_videos:
            kwargs["record_demo_videos"] = True
            if self.ui_demo_videos_dir is not None:
                kwargs["demo_videos_dir"] = self.ui_demo_videos_dir
            if self.clear_ui_demo_videos_dir:
                kwargs["demo_videos_clear"] = True

        return kwargs
