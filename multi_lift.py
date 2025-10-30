from collections import OrderedDict
import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat


class MultiCubeLift(ManipulationEnv):
    """
    A variation of the standard Lift environment that includes five cubes of different colors.
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def reward(self, action=None):
        """
        Reward function:
        - Sparse reward: +2.25 if any cube is lifted.
        - Shaped reward: reaching + grasping + lifting if enabled.
        """
        reward = 0.0

        if self._check_success():
            reward = 2.25

        elif self.reward_shaping:
            # Reaching reward: minimum distance to any cube
            dists = []
            for cube in self.cubes:
                dist = self._gripper_to_target(
                    gripper=self.robots[0].gripper,
                    target=cube.root_body,
                    target_type="body",
                    return_distance=True,
                )
                dists.append(dist)
            reaching_reward = 1 - np.tanh(10.0 * min(dists))
            reward += reaching_reward

            # Grasping reward: +0.25 if holding any cube
            for cube in self.cubes:
                if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=cube):
                    reward += 0.25
                    break

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25
        return reward

    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # ---------- Create multiple cubes ----------
        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        colors = [
            [1, 0, 0, 1],  # red
            [0, 1, 0, 1],  # green
            [0, 0, 1, 1],  # blue
            [1, 1, 0, 1],  # yellow
            [1, 0, 1, 1],  # magenta
        ]

        self.cubes = []
        for i, color in enumerate(colors):
            material = CustomMaterial(
                texture="WoodRed",
                tex_name=f"cube_tex_{i}",
                mat_name=f"cube_mat_{i}",
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib,
            )
            cube = BoxObject(
                name=f"cube{i}",
                size_min=[0.02, 0.02, 0.02],
                size_max=[0.022, 0.022, 0.022],
                rgba=color,
                material=material,
                rng=self.rng,
            )
            self.cubes.append(cube)

        # ---------- Placement sampler ----------
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            for cube in self.cubes:
                self.placement_initializer.add_objects(cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cubes,
                x_range=[-0.1, 0.1],
                y_range=[-0.1, 0.1],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )

        # ---------- Final model ----------
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cubes,
        )

    def _setup_references(self):
        super()._setup_references()
        self.cube_body_ids = {
            cube.name: self.sim.model.body_name2id(cube.root_body) for cube in self.cubes
        }

    def _setup_observables(self):
        observables = super()._setup_observables()
        if self.use_object_obs:
            modality = "object"
            sensors = []
            # cube position and orientation sensors
            for cube in self.cubes:
                @sensor(modality=modality, name=f"{cube.name}_pos")
                def cube_pos(obs_cache, cube_name=cube.name):
                    return np.array(self.sim.data.body_xpos[self.cube_body_ids[cube_name]])

                @sensor(modality=modality, name=f"{cube.name}_quat")
                def cube_quat(obs_cache, cube_name=cube.name):
                    return convert_quat(
                        np.array(self.sim.data.body_xquat[self.cube_body_ids[cube_name]]), to="xyzw"
                    )

                sensors += [cube_pos, cube_quat]

            # gripper-to-cube distances
            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])
            for cube in self.cubes:
                sensors += [
                    self._get_obj_eef_sensor(
                        full_pf, f"{cube.name}_pos", f"{arm_pf}gripper_to_{cube.name}_pos", modality
                    )
                    for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
                ]

            for s in sensors:
                observables[s.__name__] = Observable(name=s.__name__, sensor=s, sampling_rate=self.control_freq)

        return observables

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([obj_pos, obj_quat]))

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        if vis_settings.get("grippers", False):
            for cube in self.cubes:
                self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=cube)

    def _check_success(self):
        """
        Success if any cube is lifted above table + 0.04m
        """
        table_height = self.model.mujoco_arena.table_offset[2]
        for cube in self.cubes:
            cube_height = self.sim.data.body_xpos[self.cube_body_ids[cube.name]][2]
            if cube_height > table_height + 0.04:
                return True
        return False
