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
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask

from robosuite.environments.manipulation.stack import Stack
from robosuite.utils.placement_samplers import UniformRandomSampler

class StackWithCustomRandomization(Stack):
    def __init__(self, num_cubes=2, cube_colors=None, **kwargs):
        self.num_cubes = num_cubes
        self.cube_colors = cube_colors
        self.cubes = []  # Will store cube objects
        super().__init__(**kwargs)
    
    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        # Load manipulation environment (sets up robot)
        from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
        ManipulationEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Load arena (table)
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Define default colors
        if not self.cube_colors:
            self.cube_colors = [
                [1.0, 0.0, 0.0, 1.0],  # Red
                [0.0, 1.0, 0.0, 1.0],  # Green  
                [0.0, 0.0, 1.0, 1.0],  # Blue
                [1.0, 1.0, 0.0, 1.0],  # Yellow
                [1.0, 0.0, 1.0, 1.0],  # Magenta
                [0.0, 1.0, 1.0, 1.0],  # Cyan
                [1.0, 0.5, 0.0, 1.0],  # Orange
                [0.5, 0.0, 1.0, 1.0],  # Purple
                [0.5, 0.5, 0.5, 1.0],  # Gray
                [0.8, 0.4, 0.2, 1.0],  # Brown
            ]

        # Create all cubes
        self.cubes = []
        for i in range(self.num_cubes):
            cube_name = f"cube{chr(65 + i)}"  # cubeA, cubeB, cubeC, etc.
            cube = BoxObject(
                name=cube_name,
                size=[0.02, 0.02, 0.02],
                rgba=self.cube_colors[i % len(self.cube_colors)],
                obj_type="all",
                duplicate_collision_geoms=True,
            )
            self.cubes.append(cube)
        
        # Set cubeA and cubeB for compatibility with parent class
        self.cubeA = self.cubes[0]
        if len(self.cubes) > 1:
            self.cubeB = self.cubes[1]

        # Create placement initializer with custom randomization
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=self.cubes,
            x_range=[-0.20, 0.20],
            y_range=[-0.20, 0.20],
            rotation=(-np.pi/3, np.pi/3),
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        )

        # task includes arena, robot, and objects of interest
        # Use ManipulationTask to properly combine everything
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cubes,
        )
    
    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references for all cubes
        self.cube_body_ids = []
        for cube in self.cubes:
            body_id = self.sim.model.body_name2id(cube.root_body)
            self.cube_body_ids.append(body_id)


# Create environment with 5 cubes
env = StackWithCustomRandomization(
    num_cubes=5,  # Change this to add more or fewer cubes
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

env.close()
