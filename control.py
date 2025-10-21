""" 
    This is the Python file for controlling the robot using the keyboard 
    In order to have I/O, you need to run this python file as the root user
    Example: sudo ~/venvs/mj/bin/python control.py
"""

import time
import numpy as np

import robosuite as suite
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper
from robosuite.devices import Keyboard
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import BoxObject

from robosuite.environments.manipulation.stack import Stack
from robosuite.utils.placement_samplers import UniformRandomSampler

# Create environment
# env = suite.make(
    # env_name = "Stack",
    # robots="Panda",
    # has_renderer=True,
    # has_offscreen_renderer=False,
    # render_camera="agentview",
    # ignore_done=True,
    # use_camera_obs=False,
    # reward_shaping=True,
    # control_freq=20,
    # hard_reset=False,
# )

class StackWithCustomRandomization(Stack):
    def _load_model(self):
        super()._load_model()
        
        # Get the objects from the existing placement initializer
        existing_objects = self.placement_initializer.mujoco_objects
        
        # Replace with customized randomization
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=existing_objects,  # Use objects from parent
            x_range=[-0.20, 0.20],  # Wider X range
            y_range=[-0.20, 0.20],  # Wider Y range
            rotation=(-np.pi/3, np.pi/3),  # Add rotation randomization
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        )

# Then replace your suite.make() call with:
env = StackWithCustomRandomization(
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    render_camera="agentview",
    ignore_done=True,
    use_camera_obs=False,
    reward_shaping=True,
    control_freq=20,
    hard_reset=False,
)

env.reset()
# Wrap this environment in a visualization wrapper
env = VisualizationWrapper(env, indicator_configs=None)

device = Keyboard(
    env=env,
    pos_sensitivity=15,
    rot_sensitivity=15
)

while True:
        # Reset the environment
        obs = env.reset()

        # Setup rendering
        cam_id = 0
        num_cam = len(env.sim.model.camera_names)
        env.render()

        # Initialize variables that should the maintained between resets
        last_grasp = 0

        # Initialize device control
        device.start_control()
        all_prev_gripper_actions = [
            {
                f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                for robot_arm in robot.arms
                if robot.gripper[robot_arm].dof > 0
            }
            for robot in env.robots
        ]

        # Loop until we get a reset from the input or the task completes
        while True:
            start = time.time()

            # Set active robot
            active_robot = env.robots[device.active_robot]

            # Get the newest action
            input_ac_dict = device.input2action()

            # If action is none, then this a reset so we should break
            if input_ac_dict is None:
                break

            from copy import deepcopy

            action_dict = deepcopy(input_ac_dict)  # {}
            # set arm actions
            for arm in active_robot.arms:
                if isinstance(active_robot.composite_controller, WholeBody):  # input type passed to joint_action_policy
                    controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
                else:
                    controller_input_type = active_robot.part_controllers[arm].input_type

                if controller_input_type == "delta":
                    action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                elif controller_input_type == "absolute":
                    action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                else:
                    raise ValueError

            # Maintain gripper state for each robot but only update the active robot with action
            env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
            env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
            env_action = np.concatenate(env_action)
            for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]
                env.step(env_action)
                env.render()

            # limit frame rate if necessary
            # if args.max_fr is not None:
                # elapsed = time.time() - start
                # diff = 1 / args.max_fr - elapsed
                # if diff > 0:
                    # time.sleep(diff)

env.close()
