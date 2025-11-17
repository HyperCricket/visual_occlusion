import robosuite as suite
from robosuite.wrappers import VisualizationWrapper
import h5py

# Path to your robosuite demo file
path = "demo_1.hdf5"

# Launch a robosuite viewer that replays the demonstration
env = suite.make(
    "Lift",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=True,
    ignore_done=True,
    reward_shaping=True,
    control_freq=20,
)


# Wrap env to visualize each step
env = VisualizationWrapper(env)


with h5py.File(path, "r") as f:
    demo_names = [k for k in f.keys() if "demo" in k]

    for name in demo_names:
        print("Replaying:", name)
        actions = f[f"{name}/actions"][:]

        env.reset()
        for a in actions:
            env.step(a)
