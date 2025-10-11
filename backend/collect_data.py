import logging
import argparse
import sys
import time
from dataclasses import asdict
from pprint import pformat

from threading import Thread
from werkzeug.serving import make_server

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import (
    CalibrateControlConfig,
    ControlPipelineConfig,
    RecordControlConfig,
    RemoteRobotConfig,
    ReplayControlConfig,
    TeleoperateControlConfig,
)
from lerobot.common.robot_devices.control_utils import (
    init_keyboard_listener,
    record_episode_crowd,
    reset_environment_crowd,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record_crowd,
)
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import has_method, init_logging, log_say
from lerobot.configs import parser

from crowd_interface import *
from flask_app import create_flask_app
import cv2  # for closing display windows
from pathlib import Path
import os

def _stop_display_only(listener, display_cameras: bool):
    """
    Minimal UI teardown that does NOT touch the robot.
    - Listener is a daemon thread; it dies with the process.
    - Close any OpenCV windows if we showed cameras.
    """
    try:
        if display_cameras:
            cv2.destroyAllWindows()
    except Exception:
        pass

def _pop_crowd_cli_overrides(argv=None):
    """
    Extract our two lightweight CLI flags and remove them from sys.argv
    so LeRobot's config parser never sees unknown args.
    """
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--required-responses-per-critical-state", type=int, dest="crowd_rrpis")
    ap.add_argument("--autofill-critical-states", action="store_true", dest="crowd_autofill")
    ap.add_argument(
        "--num-autofill-actions",
        type=int,
        dest="crowd_num_autofill_actions",
        help="For IMPORTANT states, per unique submission fill this many actions in total "
             "(1 actual + N-1 clones). Default = required-responses-per-critical-state."
    )

    # === NEW: prompt mode flags (mutually exclusive) ===
    prompt_group = ap.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--use-vlm-prompt",
        action="store_true",
        dest="crowd_use_vlm_prompt",
        help="Use VLM-generated prompts: only serve IMPORTANT states once vlm_text and vlm_video_id exist "
             "(requires Azure OpenAI). Default if neither flag is provided is the simple task prompt."
    )
    prompt_group.add_argument(
        "--use-manual-prompt",
        action="store_true",
        dest="crowd_use_manual_prompt",
        help="Manual mode: do NOT query the VLM. Only serve IMPORTANT states after the frontend (monitor) "
             "manually sets vlm_text and vlm_video_id. Mutually exclusive with --use-vlm-prompt."
    )

    # --- NEW: save cam_main frames for critical states as an ordered image sequence ---
    ap.add_argument(
        "--save-critical-maincam-sequence",
        action="store_true",
        dest="crowd_save_seq",
        help="If set, save observation.images.cam_main at each IMPORTANT state to a directory as 000001.jpg, 000002.jpg, ...",
    )
    ap.add_argument(
        "--sequence-dir",
        type=str,
        dest="crowd_seq_dir",
        help="Directory to save IMPORTANT-state cam_main frames (default: 'prompts/demo/{--task-name}/snapshots').",
    )
    ap.add_argument(
        "--sequence-clear",
        action="store_true",
        dest="crowd_seq_clear",
        help="If set, clear the sequence directory at startup before saving new frames.",
    )
    # --- NEW: one-word task name -> derive sequence dir at <repo>/prompts/demo/{task_name} ---
    ap.add_argument(
        "--task-name",
        type=str,
        dest="crowd_task_name",
        help="One-word task name used for prompt placeholder substitution and, if --sequence-dir is not provided, "
             "to derive the sequence dir as '<repo>/prompts/demo/{task_name}/snapshots' "
             "(ignored if --sequence-dir is provided).",
    )
    # --- NEW: Leader Mode (delay VLM until N unique submissions) ---
    ap.add_argument(
        "--leader-mode",
        action="store_true",
        dest="crowd_leader_mode",
        help="Delay VLM prompt until N unique submissions for an IMPORTANT state. Requires --use-vlm-prompt."
    )
    ap.add_argument(
        "--n-leaders",
        type=int,
        dest="crowd_n_leaders",
        help="Number of unique submissions before first VLM query for an IMPORTANT state (default: 1). "
             "Must be <= --required-responses-per-critical-state."
    )
    # --- NEW: enable demo video recording in the frontend and save uploads on the backend ---
    ap.add_argument(
        "--record-demo-videos",
        action="store_true",
        dest="crowd_record_videos",
        help="If set, enable frontend demo video recording; saved under prompts/demos/{task-name}/videos.",
    )
    ap.add_argument(
        "--demo-videos-dir",
        type=str,
        dest="crowd_videos_dir",
        help="Override directory where demo videos are saved (default: 'prompts/demos/{task-name}/videos').",
    )
    ap.add_argument(
        "--auto-clear-demo",
        action="store_true",
        dest="crowd_auto_clear_demo",
        help="If set, automatically clear demo videos and snapshots directories at startup.",
    )
    ap.add_argument(
        "--show-demo-videos",
        action="store_true",
        dest="crowd_show_videos",
        help="Enable read-only example video display (no recording). Videos are read from prompts/{task-name}/videos by default."
    )
    ap.add_argument(
        "--show-videos-dir",
        type=str,
        dest="crowd_show_videos_dir",
        help="Override directory to read example videos from (default: prompts/{task-name}/videos)."
    )
    # --- NEW: save VLM conversations (context + history + current) ---
    ap.add_argument(
        "--save-vlm-logs",
        action="store_true",
        dest="crowd_save_vlm_logs",
        help="If set, save the full three-part VLM conversation per critical state to output/vlm_logs.",
    )

    args, remaining = ap.parse_known_args(argv if argv is not None else sys.argv[1:])
    # Strip our flags before LeRobot parses CLI
    sys.argv = [sys.argv[0]] + remaining
    return args

# Parse once at import time so @parser.wrap can run normally later
_CROWD_OVERRIDES = _pop_crowd_cli_overrides()

@safe_disconnect
def record(
    robot: Robot,
    crowd_interface: CrowdInterface,
    cfg: RecordControlConfig
) -> LeRobotDataset:
    if cfg.resume:
        dataset = LeRobotDataset( 
            cfg.repo_id,
            root=cfg.root,
        )
        if len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.num_image_writer_processes,
                num_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.fps, cfg.video)
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.repo_id,
            cfg.fps,
            root=cfg.root,
            robot=robot,
            use_videos=cfg.video,
            image_writer_processes=cfg.num_image_writer_processes,
            image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
        )

    crowd_interface.init_dataset(cfg, robot)

    # Load pretrained policy
    policy = make_policy(cfg.policy, ds_meta=dataset.meta) if cfg.policy is not None else None

    # Disable the leader arms since we use policy
    robot.leader_arms = []

    if not robot.is_connected:
        robot.connect()

    listener, events = init_keyboard_listener()

    # Pass events to crowd_interface for API control
    crowd_interface.set_events(events)

    # Execute a few seconds without recording to:
    # 1. teleoperate the robot to move it in starting position if no policy provided,
    # 2. give times to the robot devices to connect and start synchronizing,
    # 3. place the cameras windows on screen

    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    recorded_episodes = 0
    while True:
        if recorded_episodes >= cfg.num_episodes:
            break

        log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
        # Ensure immediate-execution only fires for submissions belonging to the
        # *currently active* episode loop.
        crowd_interface.set_active_episode(recorded_episodes)
        try:
            record_episode_crowd(
                robot=robot,
                dataset=dataset,
                events=events,
                episode_time_s=cfg.episode_time_s,
                display_cameras=cfg.display_cameras,
                policy=policy,
                fps=cfg.fps,
                single_task=cfg.single_task,
                crowd_interface=crowd_interface,
                episode_id = recorded_episodes
            )
        finally:
            # Leave no active episode once the loop exits (including early exit).
            crowd_interface.set_active_episode(None)

        # Execute a few seconds without recording to give time to manually reset the environment
        # Current code logic doesn't allow to teleoperate during this time.
        # TODO(rcadene): add an option to enable teleoperation during reset
        # Skip reset for the last episode to be recorded
        if not events["stop_recording"] and (
            (recorded_episodes < cfg.num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment", cfg.play_sounds)
            reset_environment_crowd(robot, events, cfg.reset_time_s, cfg.fps, crowd_interface)

        if events["rerecord_episode"]:
            log_say("Re-record episode", cfg.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            # Clear the crowd interface episode data
            crowd_interface.clear_episode_data(recorded_episodes)
            continue

        dataset.save_episode()
        recorded_episodes += 1

        if events["stop_recording"]:
            break
    
    crowd_interface.set_async_collection(True)  # Switch to asynchronous data collection mode
    log_say("Stop recording from cameras", cfg.play_sounds, blocking=True)
    _stop_display_only(listener, cfg.display_cameras)

    while crowd_interface.is_recording():

        log_say("Still recording from users", cfg.play_sounds, blocking=True)
        time.sleep(1)

    if cfg.push_to_hub:
        dataset.push_to_hub(tags=cfg.tags, private=cfg.private)
        crowd_interface.dataset.push_to_hub(tags=cfg.tags, private=cfg.private)


    log_say("Users have finished labeling. Exiting", cfg.play_sounds)

    return dataset
    
@parser.wrap()
def control_robot(cfg: ControlPipelineConfig):
    # exact same pattern as lerobot/scripts/control_robot.py
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Disable Flask request logging to reduce terminal noise
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

    # Wire our optional CLI overrides into the crowd interface
    ci_kwargs = {}
    if getattr(_CROWD_OVERRIDES, "crowd_rrpis", None) is not None:
        ci_kwargs["required_responses_per_critical_state"] = _CROWD_OVERRIDES.crowd_rrpis
    if getattr(_CROWD_OVERRIDES, "crowd_autofill", False):
        ci_kwargs["autofill_critical_states"] = True
    if getattr(_CROWD_OVERRIDES, "crowd_num_autofill_actions", None) is not None:
        ci_kwargs["num_autofill_actions"] = _CROWD_OVERRIDES.crowd_num_autofill_actions

    # Prompt mode selection (manual vs VLM vs simple)
    use_vlm    = bool(getattr(_CROWD_OVERRIDES, "crowd_use_vlm_prompt", False))
    use_manual = bool(getattr(_CROWD_OVERRIDES, "crowd_use_manual_prompt", False))
    if use_vlm and use_manual:
        # Redundant with argparse's mutual exclusion, but explicit here too.
        raise SystemExit("--use-manual-prompt and --use-vlm-prompt are mutually exclusive")

    if use_vlm:
        ci_kwargs["use_vlm_prompt"] = True  # existing wiring

    if use_manual:
        ci_kwargs['use_manual_prompt'] = True

    selected_mode = "manual" if use_manual else ("vlm" if use_vlm else "simple")
    logging.info(f"[Crowd] Prompt mode selected: {selected_mode}")

    # --- NEW: Leader Mode wiring and early validation ---
    if getattr(_CROWD_OVERRIDES, "crowd_leader_mode", False):
        # Enforce: only valid when either --use-vlm-prompt or --use-manual-prompt is on
        use_vlm_prompt = getattr(_CROWD_OVERRIDES, "crowd_use_vlm_prompt", False)
        use_manual_prompt = getattr(_CROWD_OVERRIDES, "crowd_use_manual_prompt", False)
        if not (use_vlm_prompt or use_manual_prompt):
            raise SystemExit("--leader-mode requires either --use-vlm-prompt or --use-manual-prompt")

        n_leaders_cli = getattr(_CROWD_OVERRIDES, "crowd_n_leaders", None)
        rrpis_cli = getattr(_CROWD_OVERRIDES, "crowd_rrpis", None)
        # If both provided, enforce the constraint up-front
        if n_leaders_cli is not None and rrpis_cli is not None and n_leaders_cli > rrpis_cli:
            raise SystemExit(
                f"--n-leaders ({n_leaders_cli}) must be <= --required-responses-per-critical-state ({rrpis_cli})"
            )

        ci_kwargs["leader_mode"] = True
        if n_leaders_cli is not None:
            ci_kwargs["n_leaders"] = n_leaders_cli

    # (rest of ci_kwargs wiring keeps its order; no behavior change)

    # NEW: image sequence saving controls
    if getattr(_CROWD_OVERRIDES, "crowd_save_seq", False):
        ci_kwargs["save_maincam_sequence"] = True
    # Always capture raw CLI task_name (if any) for prompt placeholders
    task_name = getattr(_CROWD_OVERRIDES, "crowd_task_name", None)
    safe = None
    if task_name:
        safe = "".join(c for c in task_name if (c.isalnum() or c in ("_", "-"))).strip()
        if safe:
            ci_kwargs["prompt_task_name"] = safe  # <-- used only for prompt substitution / demo assets

    # Sequence directory resolution
    if getattr(_CROWD_OVERRIDES, "crowd_seq_dir", None) is not None:
        ci_kwargs["prompt_sequence_dir"] = _CROWD_OVERRIDES.crowd_seq_dir
    else:
        # Derive prompts/{task_name}/snapshots if provided and no explicit --sequence-dir was given
        if safe:
            repo_root = Path(__file__).resolve().parent / ".."
            seq_dir = (repo_root / "prompts" / safe / "snapshots").resolve()
            ci_kwargs["prompt_sequence_dir"] = str(seq_dir)
            try:
                logging.info(f"Using derived prompt sequence dir from --task-name='{safe}': {seq_dir}")
            except Exception:
                pass
    if getattr(_CROWD_OVERRIDES, "crowd_seq_clear", False):
        ci_kwargs["prompt_sequence_clear"] = True

    # --- NEW: auto-clear control for both demo videos and snapshots ---
    auto_clear = getattr(_CROWD_OVERRIDES, "crowd_auto_clear_demo", False)
    if auto_clear:
        ci_kwargs["prompt_sequence_clear"] = True
        ci_kwargs["demo_videos_clear"] = True

    # --- NEW: demo video recording controls ---
    if getattr(_CROWD_OVERRIDES, "crowd_record_videos", False):
        ci_kwargs["record_demo_videos"] = True
        # If a custom dir isn't provided, CrowdInterface will derive prompts/{task}/videos
        if getattr(_CROWD_OVERRIDES, "crowd_videos_dir", None) is not None:
            ci_kwargs["demo_videos_dir"] = _CROWD_OVERRIDES.crowd_videos_dir
        else:
            # Derive default if we have a sanitized task name
            if safe:
                repo_root = Path(__file__).resolve().parent / ".."
                default_vdir = (repo_root / "prompts" / safe / "videos").resolve()
                ci_kwargs["demo_videos_dir"] = str(default_vdir)

    # --- NEW: read-only demo video display controls ---
    if getattr(_CROWD_OVERRIDES, "crowd_show_videos", False):
        ci_kwargs["show_demo_videos"] = True
        if getattr(_CROWD_OVERRIDES, "crowd_show_videos_dir", None) is not None:
            ci_kwargs["show_videos_dir"] = _CROWD_OVERRIDES.crowd_show_videos_dir
        else:
            if safe:
                repo_root = Path(__file__).resolve().parent / ".."
                default_show_dir = (repo_root / "prompts" / safe / "videos").resolve()
                ci_kwargs["show_videos_dir"] = str(default_show_dir)

    if getattr(_CROWD_OVERRIDES, "crowd_save_vlm_logs", False):
        ci_kwargs["save_vlm_logs"] = True

    crowd_interface = CrowdInterface(**ci_kwargs)
    crowd_interface.init_cameras()

    app = create_flask_app(crowd_interface)
    # Use a controllable WSGI server instead of app.run() in a daemon thread.
    http_server = make_server("0.0.0.0", 9000, app)
    server_thread = Thread(target=http_server.serve_forever, name="flask-wsgi", daemon=True)
    server_thread.start()

    robot = make_robot_from_config(cfg.robot)

    assert isinstance(cfg.control, RecordControlConfig), 'This script is for data collection'

    record(robot, crowd_interface, cfg.control)

    if robot.is_connected:
        robot.disconnect()

    print('Data Collection Completed')


if __name__ == "__main__":
    control_robot()