# In your main robot control script
from threading import Thread
from crowd_interface import CrowdInterface, create_flask_app, JOINT_NAMES
import trossen_arm
import numpy as np

from test import dummy

import time

def robot_loop_with_crowd_interface():
    # Initialize the crowd interface
    crowd_interface = CrowdInterface()
    crowd_interface.init_cameras()
    
    # Create and start Flask app in background
    app = create_flask_app(crowd_interface)
    server_thread = Thread(
        target=lambda: app.run(host="0.0.0.0", port=9000, debug=False, use_reloader=False),
        daemon=True
    )
    server_thread.start()

    dummy()
    
    # Initialize robot
    driver = trossen_arm.TrossenArmDriver()
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_leader,
        '192.168.1.3',
        True
    )
    
    convert_rad = np.pi / 180.0
    driver.set_all_modes(trossen_arm.Mode.position)
    driver.set_all_positions(
        np.array([0.0, 60.0, 75.0, -60.0, 0.0, 0.0, 2.0]) * convert_rad,
        2.0, True
    )
    
    gripper_motion = 1

    time.sleep(2.0)

    vec = list(driver.get_all_positions())
    joint_map = {n: v for n, v in zip(JOINT_NAMES, vec)}
    
    # Add state to crowd interface
    crowd_interface.add_state(joint_map, gripper_motion)
    
    past_goal_positions = vec  # Initialize with current position
    past_gripper_action = gripper_motion
    
    try:
        while True:
            # Replicate teleop_step_crowd logic exactly
            
            # Read velocity to check if moving (simplified - assume we have velocity info)
            velocity = driver.get_all_velocities() if hasattr(driver, 'get_all_velocities') else [0.0] * 7
            moving = abs(np.max(velocity) - 0) > 0.1
            
            # add state if we are moving
            if moving:
                present_position = list(driver.get_all_positions())
                joint_map = {n: v for n, v in zip(JOINT_NAMES, present_position)}
                crowd_interface.add_state(joint_map, past_gripper_action)
            
            # get goal only when we stop
            goal = crowd_interface.get_latest_goal()

            if goal:
                print(f'quick_teleoperation received: {goal}')

            goal_positions = past_goal_positions
            gripper_action = past_gripper_action

            if goal:
                goal_positions = [goal['joint_positions'][key][0] for key in JOINT_NAMES]
                gripper_action = float(goal['gripper'])
            
            goal_positions = np.array(goal_positions)
            gripper_action = np.array([gripper_action])

            past_goal_positions = goal_positions

            goal_joint_action = np.concat((goal_positions, gripper_action))

            # Send the goal to robot (equivalent to write("Goal_Joint_Action"))
            driver.set_all_positions(goal_positions, 2.0, False)
            
            time.sleep(1/30)  # 30 FPS
            
    except KeyboardInterrupt:
        pass
    finally:
        crowd_interface.cleanup_cameras()
        # Reset robot to safe position
        driver.set_all_positions(np.zeros(driver.get_num_joints()), 2.0, True)

if __name__ == '__main__':
    robot_loop_with_crowd_interface()