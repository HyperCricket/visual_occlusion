import numpy as np
import torch

from control import StackWithCustomRandomization   # assumes control.py is safe to import
from robosuite.wrappers import VisualizationWrapper

# ============================================================
# Device setup
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# Observation helper (matches data collection)
# ============================================================

def get_obs_vector(env, robot, arm_name="right"):
    """
    Build the same observation vector you stored in HDF5 demos:

        [joint_pos, joint_vel, eef_pos, eef_quat, gripper_qpos]
    """
    eef_site_id = robot.sim.model.site_name2id("gripper0_right_grip_site")
    eef_body_name = "robot0_right_hand"

    joint_pos = robot.sim.data.qpos[robot._ref_joint_pos_indexes].copy()
    joint_vel = robot.sim.data.qvel[robot._ref_joint_vel_indexes].copy()
    eef_pos = robot.sim.data.site_xpos[eef_site_id].copy()
    eef_quat = robot.sim.data.get_body_xquat(eef_body_name).copy()
    gripper_qpos = robot.sim.data.qpos[robot._ref_gripper_joint_pos_indexes[arm_name]].copy()

    obs_vec = np.concatenate([joint_pos, joint_vel, eef_pos, eef_quat, gripper_qpos])
    return obs_vec


# ============================================================
# Environment creation
# ============================================================

def make_env():
    """
    Create the same environment configuration you used for data collection.
    """
    env = StackWithCustomRandomization(
        num_cubes=10,
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="frontview",
        ignore_done=True,      # set to False if you want episodes to actually end
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
    )
    env = VisualizationWrapper(env, indicator_configs=None)
    return env


# ============================================================
# Policy loading
# ============================================================

def build_policy_from_state_dict(state_dict):
    """
    Construct your DiffusionPolicy model and load the given state_dict.

    !!! IMPORTANT !!!
    You MUST modify this function to match how you constructed your policy
    during training.

    Example (you must change this to your real constructor):

        from my_model_file import DiffusionPolicy

        obs_dim = state_dict["some_layer.weight"].shape[1]  # or set manually
        action_dim = ...

        policy = DiffusionPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            horizon=16,
            hidden_dim=256,
            ...
        )

        policy.load_state_dict(state_dict)
        return policy
    """
    # ======= TODO: EDIT THIS TO MATCH YOUR TRAINING CODE =======
    # Placeholder that will crash loudly if you forget to edit it:
    raise NotImplementedError(
        "You must implement build_policy_from_state_dict(state_dict) "
        "using your actual DiffusionPolicy class and constructor."
    )
    # ===========================================================


def load_policy(path="diffusion_policy.pth"):
    """
    Load the diffusion policy from a .pth file.

    Supports:
      - Full model saved with `torch.save(model, path)`
      - state_dict saved with `torch.save(model.state_dict(), path)`
      - checkpoint dict saved with `torch.save({'model': state_dict, ...}, path)`
    """
    raw = torch.load(path, map_location=device)
    print(f"Loaded raw checkpoint of type: {type(raw)}")

    # Case 1: full model was saved directly
    if isinstance(raw, torch.nn.Module):
        policy = raw
        policy.to(device)
        policy.eval()
        print("Loaded full model from", path)
        return policy

    # Case 2: checkpoint dict
    if isinstance(raw, dict):
        # Common patterns: {"model": state_dict, ...} or pure state_dict
        if any(isinstance(v, torch.Tensor) for v in raw.values()):
            # Likely a pure state_dict or similar
            state_dict = raw
            print("Interpreting checkpoint as pure state_dict with keys:", list(state_dict.keys())[:5], "...")
        elif "model" in raw:
            state_dict = raw["model"]
            print("Found 'model' key in checkpoint dict; using that as state_dict")
        else:
            raise RuntimeError(
                "Checkpoint dict format not recognized. "
                "Expected either a pure state_dict or a dict with key 'model'. "
                f"Got keys: {list(raw.keys())}"
            )

        policy = build_policy_from_state_dict(state_dict)
        policy.to(device)
        policy.eval()
        print("Constructed policy from state_dict in", path)
        return policy

    # Case 3: pure OrderedDict state_dict
    from collections import OrderedDict
    if isinstance(raw, OrderedDict):
        print("Checkpoint is an OrderedDict state_dict with keys:", list(raw.keys())[:5], "...")
        policy = build_policy_from_state_dict(raw)
        policy.to(device)
        policy.eval()
        print("Constructed policy from OrderedDict state_dict in", path)
        return policy

    # If we reach here, we don't know what this is
    raise RuntimeError(
        f"Unsupported checkpoint type {type(raw)}. "
        "Expected nn.Module, dict, or OrderedDict."
    )


# ============================================================
# Rollout loop
# ============================================================

def run_policy_rollout(policy, num_episodes=3, max_steps=300, history_len=1):
    """
    Rollout the trained policy in robosuite and render.

    history_len:
      - 1 if your policy takes a single observation (shape: [B, D])
      - >1 if your policy takes a sequence (shape: [B, H, D])
    """
    env = make_env()
    robot = env.robots[0]

    try:
        for ep in range(num_episodes):
            obs = env.reset()

            # Same neutral pose as in data collection
            neutral_joints = np.array([0, -0.3, 0, -2.0, 0, 1.7, 0.785])
            noise = np.random.uniform(-0.2, 0.2, size=7)
            robot.set_robot_joint_positions(neutral_joints + noise)
            env.sim.forward()

            print(f"\n=== Episode {ep + 1} ===")
            obs_vec = get_obs_vector(env, robot, arm_name="right")

            obs_history = []

            for t in range(max_steps):
                # Maintain history for diffusion models that use context
                obs_history.append(obs_vec)
                obs_history = obs_history[-history_len:]

                if history_len > 1:
                    # Shape (1, H, D)
                    obs_input = np.stack(obs_history, axis=0)[None, ...]
                else:
                    # Shape (1, D)
                    obs_input = obs_vec[None, :]

                obs_tensor = torch.from_numpy(obs_input).float().to(device)

                # Query policy
                with torch.no_grad():
                    action = policy(obs_tensor)

                # If policy returns (action, extra_info), just take action
                if isinstance(action, (tuple, list)):
                    action = action[0]

                action_np = action.detach().cpu().numpy()[0]

                # Optional: debug check that actions are non-zero
                # print(f"step {t}: action norm = {np.linalg.norm(action_np):.4f}")

                # Step environment
                obs, reward, done, info = env.step(action_np)
                env.render()

                # Update observation for next step
                obs_vec = get_obs_vector(env, robot, arm_name="right")

                if not env.ignore_done and done:
                    print(f"Episode ended at step {t + 1} with reward {reward:.3f}")
                    break

        env.close()

    except KeyboardInterrupt:
        print("\nInterrupted by user; closing env.")
        env.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    policy = load_policy("diffusion_policy.pth")
    # Set history_len to your diffusion context length if you used >1
    run_policy_rollout(policy, num_episodes=5, max_steps=400, history_len=1)

