""" 
    This is the Python file for controlling the robot using the keyboard, used to collect demonstrations in HDF5 format.
    In order to have I/O, you need to run this python file as the root user
    Example: sudo ~/venvs/mj/bin/python control.py
"""
import time
import numpy as np
import h5py
from datetime import datetime
import robosuite as suite
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper
from robosuite.devices import Keyboard
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import BoxObject
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.environments.manipulation.stack import Stack
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.models.objects import BoxObject

import numpy as np  # make sure this is at the top of the file

def stack_cubeB_on_cubeA(env, obs):
    """
    After env.reset(), move cubeB so it is stacked on top of cubeA.

    - Uses obs["cubeA_pos"] for cubeA's location
    - Uses MuJoCo to find cubeB's free joint and rewrites its qpos
    - Works even if env is wrapped in VisualizationWrapper
    """
    # If env is a wrapper (VisualizationWrapper), unwrap to get base env
    base_env = getattr(env, "env", env)
    sim = base_env.sim

    # 1) Get cubeA position from observations (already correct)
    cubeA_pos = np.array(obs["cubeA_pos"])   # shape (3,)

    # 2) Compute desired cubeB center: directly above cubeA
    cube_half = 0.02  # BoxObject size=[0.02,0.02,0.02] -> half side
    top_center = cubeA_pos.copy()
    top_center[2] = cubeA_pos[2] + 2.0 * cube_half  # one cube height above

    # 3) Find cubeB body (note: names are "cubeA_main", "cubeB_main", ...)
    try:
        cubeA_body_id = sim.model.body_name2id("cubeA_main")
        cubeB_body_id = sim.model.body_name2id("cubeB_main")
    except Exception as e:
        print("ERROR: could not find cubeA_main / cubeB_main bodies:", e)
        print("Available bodies:", [sim.model.body_id2name(i) for i in range(sim.model.nbody)])
        return

    # 4) Find cubeB's joint and its qpos slice
    body_jnt_adr = sim.model.body_jntadr[cubeB_body_id]
    body_jnt_num = sim.model.body_jntnum[cubeB_body_id]
    if body_jnt_num == 0:
        print("WARNING: cubeB_main has no joints, cannot move it via qpos.")
        return

    joint_id = int(body_jnt_adr)  # first joint attached to cubeB_main
    joint_name = sim.model.joint_id2name(joint_id)

    addr = sim.model.get_joint_qpos_addr(joint_name)
    if isinstance(addr, (tuple, list, np.ndarray)):
        start, end = int(addr[0]), int(addr[1])
    else:
        start, end = int(addr), int(addr) + 7  # free joint -> 7 dof (xyz + quat)

    sl = slice(start, end)

    # 5) Rewrite cubeB free joint qpos
    qpos = np.array(sim.data.qpos[sl], dtype=np.float64)
    if qpos.shape[0] != 7:
        print("WARNING: cubeB joint qpos has unexpected shape:", qpos.shape)
        return

    qpos[0:3] = top_center
    qpos[3:] = np.array([1.0, 0.0, 0.0, 0.0])  # identity orientation

    sim.data.qpos[sl] = qpos
    sim.forward()

    # 6) Debug print
    print("cubeA body_xpos:", sim.data.body_xpos[cubeA_body_id])
    print("cubeB body_xpos (stacked):", sim.data.body_xpos[cubeB_body_id])

class StackWithCustomRandomization(Stack):
    def __init__(self, num_cubes=2, cube_colors=None, **kwargs):
        self.num_cubes = num_cubes
        self.cube_colors = cube_colors
        self.cubes = []
        self.cube_body_ids = []
        super().__init__(**kwargs)
    
    def _load_model(self):
        ManipulationEnv._load_model(self)

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # ---------- COLORS: only cubeA is red ----------
        # cubeA: red
        red = [1.0, 0.0, 0.0, 1.0]

        # other cubes: anything BUT red
        non_red_colors = [
            [0.0, 1.0, 0.0, 1.0],  # green
            [0.0, 0.0, 1.0, 1.0],  # blue
            [1.0, 1.0, 0.0, 1.0],  # yellow
            [1.0, 0.0, 1.0, 1.0],  # magenta
            [0.0, 1.0, 1.0, 1.0],  # cyan
            [1.0, 0.5, 0.0, 1.0],  # orange
            [0.5, 0.0, 1.0, 1.0],  # purple
            [0.5, 0.5, 0.5, 1.0],  # gray
            [0.8, 0.4, 0.2, 1.0],  # brown
        ]
        # ----------------------------------------------

        self.cubes = []
        for i in range(self.num_cubes):
            cube_name = f"cube{chr(65 + i)}"  # "cubeA", "cubeB", ...

            if i == 0:
                rgba = red                      # cubeA always red
            else:
                rgba = non_red_colors[(i - 1) % len(non_red_colors)]  # others non-red

            cube = BoxObject(
                name=cube_name,
                size=[0.02, 0.02, 0.02],
                rgba=rgba,
                obj_type="all",
                duplicate_collision_geoms=True,
            )
            self.cubes.append(cube)

        self.cubeA = self.cubes[0]
        if len(self.cubes) > 1:
            self.cubeB = self.cubes[1]

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

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cubes,
        )

    def _setup_references(self):
        super()._setup_references()
        self.cube_body_ids = []
        for cube in self.cubes:
            body_id = self.sim.model.body_name2id(cube.root_body)
            self.cube_body_ids.append(body_id)

def main():
    env = StackWithCustomRandomization(
        num_cubes=10,
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="frontview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=15,
        hard_reset=False,
    )

    env.reset()
    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    device = Keyboard(
        env=env,
        pos_sensitivity=8,
        rot_sensitivity=8
    )

    # ==== DATA COLLECTION SETUP ====
    # Create HDF5 file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hdf5_filename = f"demonstrations_{timestamp}.hdf5"
    hdf5_file = h5py.File(hdf5_filename, 'w')

    # Metadata
    hdf5_file.attrs['date'] = timestamp
    hdf5_file.attrs['env_name'] = 'StackWithCustomRandomization'
    hdf5_file.attrs['num_cubes'] = 2 

    demo_counter = 0
    print(f"Recording demonstrations to: {hdf5_filename}")
    print("Press 'Q' to reset and start a new demonstration")
    print("Each episode will be saved automatically")
    # ================================

    try:
        while True:
            # Reset the environment
            obs = env.reset()
            print("cubeA_pos:", obs["cubeA_pos"])
            env.render()

            stack_cubeB_on_cubeA(env, obs) 

            # RANDOMIZE JOINT POSITIONS (hand stays visible and near table center)
            robot = env.robots[0]
            
            # Set to a good neutral hovering position that's visible
            neutral_joints = np.array([0, -0.3, 0, -2.0, 0, 1.7, 0.785])
            
            # Add small random variations to each joint
            noise = np.random.uniform(-0.2, 0.2, size=7)
            
            robot.set_robot_joint_positions(neutral_joints + noise)
            env.sim.forward()
            obs = env._get_observations()
            # END RANDOMIZATION
            
            # Setup rendering
            cam_id = 0
            num_cam = len(env.sim.model.camera_names)
            env.render()
            
            # ==== START NEW DEMONSTRATION ====
            demo_counter += 1
            demo_group = hdf5_file.create_group(f"demo_{demo_counter}")

            # Lists to store episode data
            actions_list = []
            observations_list = []
            rewards_list = []
            dones_list = []

            # Store initial joint positions
            demo_group.attrs['initial_joint_positions'] = neutral_joints + noise

            # Define arm name
            arm_name = 'right'

            # Get end-effector site ID - use the grip site
            eef_site_id = robot.sim.model.site_name2id('gripper0_right_grip_site')

            # Get the body name for the quaternion
            eef_body_name = 'robot0_right_hand'

            print(f"\n=== Starting Demo {demo_counter} ===")
            step_counter = 0
            # =================================

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
                
                # Step environment
                obs, reward, done, info = env.step(env_action)
                actions_list.append(env_action)

                # print("env_action shape:", env_action.shape)

                obs_to_save = {
                    "robot0_joint_pos": obs["robot0_joint_pos"],
                    "robot0_joint_vel": obs["robot0_joint_vel"],
                    "robot0_gripper_qpos": obs["robot0_gripper_qpos"],
                    "robot0_eef_pos": obs["robot0_eef_pos"],
                    "robot0_eef_quat": obs["robot0_eef_quat"],
                    "cubeA_pos": obs["cubeA_pos"],
                    "cubeA_quat": obs["cubeA_quat"],
                    "cubeB_pos": obs["cubeB_pos"],
                    "cubeB_quat": obs["cubeB_quat"],
                    "gripper_to_cubeA": obs["gripper_to_cubeA"],
                    "gripper_to_cubeB": obs["gripper_to_cubeB"],
                    # or just obs["object-state"] if you want the bundled one
                }
                observations_list.append(obs_to_save)
                rewards_list.append(reward)
                dones_list.append(done)

                # =====================
                step_counter += 1

                env.render()
            
            # ==== SAVE DEMONSTRATION TO HDF5 ====
            # print("num obs:", len(observations_list))
            # print("num actions:", len(actions_list))
            # if len(actions_list) > 0:
                # print("first action shape:", actions_list[0].shape)

            print(f"Demo {demo_counter} finished with {step_counter} steps. Saving...")
            
            # Save actions
            demo_group.create_dataset('actions', data=np.array(actions_list))
            
            # Save observations (create subgroup for observations)
            obs_group = demo_group.create_group('observations')

            obs_group.create_dataset(
                'robot0_joint_pos',
                data=np.array([o['robot0_joint_pos'] for o in observations_list])
            )
            obs_group.create_dataset(
                'robot0_joint_vel',
                data=np.array([o['robot0_joint_vel'] for o in observations_list])
            )
            obs_group.create_dataset(
                'robot0_gripper_qpos',
                data=np.array([o['robot0_gripper_qpos'] for o in observations_list])
            )

            obs_group.create_dataset(
                'robot0_eef_pos',
                data=np.array([o['robot0_eef_pos'] for o in observations_list])
            )
            obs_group.create_dataset(
                'robot0_eef_quat',
                data=np.array([o['robot0_eef_quat'] for o in observations_list])
            )

            obs_group.create_dataset(
                'cubeA_pos',
                data=np.array([o['cubeA_pos'] for o in observations_list])
            )
            obs_group.create_dataset(
                'cubeA_quat',
                data=np.array([o['cubeA_quat'] for o in observations_list])
            )
            obs_group.create_dataset(
                'cubeB_pos',
                data=np.array([o['cubeB_pos'] for o in observations_list])
            )
            obs_group.create_dataset(
                'cubeB_quat',
                data=np.array([o['cubeB_quat'] for o in observations_list])
            )

            obs_group.create_dataset(
                'gripper_to_cubeA',
                data=np.array([o['gripper_to_cubeA'] for o in observations_list])
            )
            obs_group.create_dataset(
                'gripper_to_cubeB',
                data=np.array([o['gripper_to_cubeB'] for o in observations_list])
            )

            # Save rewards and dones
            demo_group.create_dataset('rewards', data=np.array(rewards_list))
            demo_group.create_dataset('dones', data=np.array(dones_list))
            
            # Save metadata
            demo_group.attrs['num_steps'] = step_counter
            demo_group.attrs['total_reward'] = np.sum(rewards_list)
            
            print(f"Saved! Total reward: {np.sum(rewards_list):.2f}")
            
            # Flush to disk for safety
            hdf5_file.flush()

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected -- stopping gracefully...")


    finally:
        # Ensure cleanup no matter how program ends
        hdf5_file.close()
        env.close()
        print(f"\nAll demonstrations saved to: {hdf5_filename}")
        print("Environment closed cleanly.")
        # ====================================

    # Close HDF5 file when done (this won't run until you exit the program)
    hdf5_file.close()
    print(f"\nAll demonstrations saved to: {hdf5_filename}")
    env.close()

if __name__ == "__main__":
    main()
