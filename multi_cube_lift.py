from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler


class MultiCubeLift(Lift):
    def _load_model(self):
        # Load everything from Lift first (arena, robot, table, etc.)
        super()._load_model()

        # Define multiple cubes
        colors = [
            [1, 0, 0, 1],  # red (target)
            [0, 1, 0, 1],  # green
            [0, 0, 1, 1],  # blue
            [1, 1, 0, 1],  # yellow
            [1, 0, 1, 1],  # magenta
        ]

        self.cubes = []
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        for i, color in enumerate(colors):
            cube = BoxObject(
                name=f"cube_{i}",
                size_min=[0.02, 0.02, 0.02],
                size_max=[0.02, 0.02, 0.02],
                rgba=color,
                material=redwood,
                rng=self.rng,
            )
            self.cubes.append(cube)

        self.target_cube = self.cubes[0]  # red = success cube

        # Create new placement initializer to place all cubes randomly
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=self.cubes,
            x_range=[-0.15, 0.15],
            y_range=[-0.15, 0.15],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
            rng=self.rng,
        )

        # Build a new full task model with all cubes
        self.model = ManipulationTask(
            mujoco_arena=self.model.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cubes,
        )

    def _setup_references(self):
        # Call Lift setup first (sets up robot references)
        super()._setup_references()
        # Add reference for red target cube
        self.target_cube_body_id = self.sim.model.body_name2id(self.target_cube.root_body)

    def _check_success(self):
        # Only success if red cube is lifted
        cube_height = self.sim.data.body_xpos[self.target_cube_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]
        return cube_height > table_height + 0.04
