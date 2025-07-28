
import random, json, asyncio, websockets
import trossen_arm
import numpy as np

### CONFIGS--- ###

num_trajectories = 1
num_goal_poses_each_step = 1

### ---CONFIGS ###


def initialize_driver():
    server_ip = '192.168.1.2'
    driver = trossen_arm.TrossenArmDriver()
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_leader,
        server_ip,
        False
    )

    driver.set_all_modes(trossen_arm.Mode.position)

    return driver

def initialize_arm(driver):
    driver.set_all_positions(
    np.array([0.0, 60.0, 75.0, -60.0, 0.0, 0.0, 2.0]) * np.pi / 180.0,
    2.0,
    True
)

def get_human_input(num_goal_poses_each_step, driver):

    # not implemented
    pass



def collect_data(driver):

    # policy = initialize_policy()

    # data collection loop
    for i in range(num_trajectories):
        data = [] # list of dictionaries

        initialize_arm(driver)

        try: # interrupt traj with ctrl-C
            while True:
                goal_poses = get_human_input(num_goal_poses_each_step, driver)

                goal_pose = random.choice(goal_poses)

                driver.set_cartesian_positions(
                    goal_pose,
                    trossen_arm.InterpolationSpace.cartesian,
                    goal_time = 2.0,
                    blocking = False
                )

        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    driver = initialize_driver()
    initialize_arm(driver)
    collect_data(driver)