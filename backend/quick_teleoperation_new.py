# In your main robot control script
from threading import Thread
from crowd_interface import CrowdInterface, create_flask_app, JOINT_NAMES
import trossen_arm
import numpy as np

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
    
    try:
        while True:
            # Get current robot state
            vec = list(driver.get_all_positions())
            joint_map = {n: v for n, v in zip(JOINT_NAMES, vec)}
            
            # Add state to crowd interface
            crowd_interface.add_state(joint_map, gripper_motion)
            
            # Check for new goals from frontend
            goal = crowd_interface.get_latest_goal()
            if goal:
                # Process the goal
                goal_positions = np.array([
                    goal["joint_positions"][n] for n in JOINT_NAMES
                ], dtype=float)
                
                driver.set_all_positions(goal_positions, 2.0, False)
                
                # Handle gripper
                if goal.get("gripper") in (-1, +1):
                    target = 0.044 if goal["gripper"] > 0 else 0.000
                    driver.set_gripper_position(target, 0.0, False)
                    gripper_motion = goal["gripper"]
            
            time.sleep(1/30)  # 30 FPS
            
    except KeyboardInterrupt:
        pass
    finally:
        crowd_interface.cleanup_cameras()
        # Reset robot to safe position
        driver.set_all_positions(np.zeros(driver.get_num_joints()), 2.0, True)

if __name__ == '__main__':
    robot_loop_with_crowd_interface()