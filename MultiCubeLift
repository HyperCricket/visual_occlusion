from robosuite.environments.manipulation.lift import Lift

class MultiCubeLift(Lift):
    def _load_model(self):
        super()._load_model()

        # Add multiple cubes
        from robosuite.models.objects import BoxObject
        import numpy as np

        colors = [
            [1, 0, 0, 1],  # red (target)
            [0, 1, 0, 1],  # green
            [0, 0, 1, 1],  # blue
            [1, 1, 0, 1],  # yellow
            [1, 0, 1, 1],  # magenta
        ]

        self.cubes = []
        for i, color in enumerate(colors):
            cube = BoxObject(
                name=f"cube_{i}",
                size_min=[0.02, 0.02, 0.02],
                size_max=[0.02, 0.02, 0.02],
                rgba=color,
            )
            self.cubes.append(cube)
            self.model.merge(cube.get_model(mode="mujoco"))

        self.target_cube = self.cubes[0]  # red one = target

    def _setup_references(self):
        super()._setup_references()
        self.target_cube_body_id = self.sim.model.body_name2id(self.target_cube.root_body)

    def _check_success(self):
        # Only count success if red cube (target) is lifted
        cube_height = self.sim.data.body_xpos[self.target_cube_body_id][2]
        return cube_height > self.table_offset[2] + 0.04
