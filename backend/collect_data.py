import logging
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
    ControlPipelineConfig,
    RecordControlConfig,
)
from lerobot.common.robot_devices.control_utils import (
    init_keyboard_listener,
    record_episode_crowd,
    reset_environment_crowd,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import safe_disconnect
from lerobot.common.utils.utils import has_method, init_logging, log_say
from lerobot.configs import parser

from crowd_interface import *
from crowd_interface_config import CrowdInterfaceConfig
from flask_app import create_flask_app
import cv2  # for closing display windows
from pathlib import Path

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

# Parse crowd interface config once at import time so @parser.wrap can run normally later
_CROWD_CONFIG = CrowdInterfaceConfig.from_cli_args()

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
            # TODO: Implement clear_episode_data in CrowdInterface if needed
            # crowd_interface.clear_episode_data(recorded_episodes)
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

    # Log the prompt mode being used
    prompt_mode = "manual" if _CROWD_CONFIG.use_manual_prompt else "simple"
    logging.info(f"[Crowd] Prompt mode selected: {prompt_mode}")
    logging.info(f"[Crowd] Task name: {_CROWD_CONFIG.task_name}")

    # Use the crowd interface config to create CrowdInterface
    crowd_interface = CrowdInterface(**_CROWD_CONFIG.to_crowd_interface_kwargs())
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