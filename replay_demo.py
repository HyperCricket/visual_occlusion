import h5py
import numpy as np

from control import StackWithCustomRandomization
from robosuite.wrappers import VisualizationWrapper


def make_env():
    env = StackWithCustomRandomization(
        num_cubes=2,
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
    env = VisualizationWrapper(env, indicator_configs=None)
    return env


if __name__ == "__main__":
    hdf5_path = "demonstrations_20251126_225141.hdf5"   # <- use your actual file
    demo_key = "demo_1"                                 # or "demo_42", etc.

    # ----- Load actions + initial joint positions from HDF5 -----
    with h5py.File(hdf5_path, "r") as f:
        actions = np.array(f[f"{demo_key}/actions"])  # (T, 7)
        init_qpos = f[f"{demo_key}"].attrs["initial_joint_positions"]

    print(f"Replaying {demo_key}: {actions.shape[0]} steps")

    # ----- Make env and roughly restore robot pose -----
    env = make_env()
    obs = env.reset()

    robot = env.robots[0]
    robot.set_robot_joint_positions(init_qpos)
    env.sim.forward()
    env.render()

    # ----- Step through recorded actions -----
    for t, action in enumerate(actions):
        if t < 5:
            print(f"action[{t}]:", action)

        obs, reward, done, info = env.step(action)
        env.render()

    env.close()
    print("Replay finished.")

