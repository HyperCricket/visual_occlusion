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

with h5py.File(path, "r") as f:
    demo = f["demo_1"]
    actions = demo["actions"][:]

# Wrap env to visualize each step
env = VisualizationWrapper(env)

for a in actions:
    env.step(a)
