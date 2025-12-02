import math
import numpy as np
import torch
import torch.nn as nn

from control import StackWithCustomRandomization
from robosuite.wrappers import VisualizationWrapper
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from train import ConditionalUnet1D

# ============================================================
# Device setup
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# Observation helper (low-dim, matches our training)
# ============================================================

def get_obs_vector(obs):
    base = np.concatenate([
        obs["robot0_joint_pos"],        # (7,)
        obs["robot0_joint_vel"],        # (7,)
        obs["robot0_gripper_qpos"],     # (2,)
        obs["cubeA_pos"],               # (3,)
        obs["cubeB_pos"],               # (3,)
        obs["gripper_to_cubeA"],        # (3,)
        obs["gripper_to_cubeB"],        # (3,)
    ]).astype(np.float32)

    # unpack
    joint_pos = base[0:7]
    joint_vel = base[7:14]
    gripper_qpos = base[14:16]
    cubeA_pos = base[16:19]
    cubeB_pos = base[19:22]
    g2A = base[22:25]
    g2B = base[25:28]

    dist_to_A = np.linalg.norm(g2A) + 1e-8
    gripper_closed = float(gripper_qpos.mean() > 0.01)
    eef_above_A = float((-g2A[2]) > 0.03)

    phase = 0.0
    if dist_to_A < 0.10 and gripper_closed < 0.5:
        phase = 1.0
    if dist_to_A < 0.08 and gripper_closed > 0.5 and eef_above_A > 0.5:
        phase = 2.0

    return np.concatenate([base, np.array([phase], dtype=np.float32)])
 
# ============================================================
# Environment creation
# ============================================================
 
def make_env():
    """
    Create the same environment configuration you used for data collection.
    """
    env = StackWithCustomRandomization(
        num_cubes=2,
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="frontview",
        ignore_done=True,      # set to False if you want episodes to actually end
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=15,
        hard_reset=False,
    )
    env = VisualizationWrapper(env, indicator_configs=None)
    return env

# ============================================================
# Loading the trained diffusion policy
# ============================================================

OBS_DIM = 29    
ACTION_DIM = 7
OBS_HORIZON = 16
NUM_DIFFUSION_ITERS = 1000
MODEL_PATH = "/home/kevin/Programming/Research/visual_occlusion/experiments/lift_diffusion_image_v15/diffusion_lift_image_v15/20251201110550/last.pth"   # change if you used a different filename

def load_diffusion_policy(path=MODEL_PATH, device=device):
    """
    Recreate the diffusion model architecture and load the state_dict.
    Also create a matching DDPM scheduler for sampling.
    """
    model = ConditionalUnet1D(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        horizon=OBS_HORIZON
    ).to(device)

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    scheduler = DDPMScheduler(
        num_train_timesteps=NUM_DIFFUSION_ITERS,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    print(f"Loaded diffusion policy from {path}")
    return model, scheduler

# ============================================================
# Sampling: obs history -> one action
# ============================================================

@torch.no_grad()
def sample_action_from_history(
    model,
    scheduler,
    obs_history,                  # list of np arrays, each (OBS_DIM,)
    obs_horizon=OBS_HORIZON,
    num_inference_steps=50,
    device=device
):
    """
    Turn a history of low-dim observations into one 7D action using reverse diffusion.

    obs_history: list of obs vectors (np.ndarray (16,))
    """
    # Keep only the last obs_horizon steps, pad if too short
    if len(obs_history) >= obs_horizon:
        recent = obs_history[-obs_horizon:]
    else:
        # pad at the beginning by repeating the first element
        pad_count = obs_horizon - len(obs_history)
        recent = [obs_history[0]] * pad_count + obs_history

    obs_seq = np.stack(recent, axis=0)     # (H, 16)
    obs_flat = obs_seq.flatten().astype(np.float32)  # (H*16,)

    obs_cond = torch.from_numpy(obs_flat).unsqueeze(0).to(device)  # (1, H*16)

    # Set up timesteps for inference
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps  # e.g. [999, 979, ..., 0]

    # Start from pure Gaussian noise as the action
    action = torch.randn((1, ACTION_DIM), device=device)

    for t in timesteps:
        t_batch = torch.tensor([t], device=device).long()  # (1,)
        noise_pred = model(action, t_batch, obs_cond)      # (1, 7)

        step = scheduler.step(noise_pred, t, action)
        action = step.prev_sample

    # x_0 is our final sampled action
    return action.squeeze(0).cpu().numpy()   # (7,)

# ============================================================
# Rollout loop
# ============================================================

def run_policy_rollout(model, scheduler, num_episodes=3, max_steps=300):
    """
    Rollout the trained diffusion policy in robosuite and render.
    """
    env = make_env()
    robot = env.robots[0]

    try:
        for ep in range(num_episodes):
            obs = env.reset()
            print("obs keys at reset:", obs.keys())

            # Same neutral pose as in data collection (tweak as needed)
            neutral_joints = np.array([0, -0.3, 0, -2.0, 0, 1.7, 0.785])
            noise = np.random.uniform(-0.2, 0.2, size=7)
            robot.set_robot_joint_positions(neutral_joints + noise)
            env.sim.forward()

            obs = env._get_observations()
            print(f"\n=== Episode {ep + 1} ===")

            # Initial obs
            obs_vec = get_obs_vector(obs)
            obs_history = [obs_vec]

            for t in range(max_steps):
                # Sample an action from the diffusion policy
                action_np = sample_action_from_history(
                    model,
                    scheduler,
                    obs_history,
                    obs_horizon=OBS_HORIZON,
                    num_inference_steps=50,
                    device=device
                )

                # print(action_np.shape)

                # Step environment
                print("rollout action:", action_np, "norm:", np.linalg.norm(action_np))
                # obs, reward, done, info = env.step(action_np)
                SCALE_JOINTS = 0.5
                SCALE_GRIPPER = 1.0

                env_action = action_np.copy()
                env_action[:6] *= SCALE_JOINTS
                env_action[6]  *= SCALE_GRIPPER
                env_action = np.clip(env_action, -1.0, 1.0)

                obs, reward, done, info = env.step(env_action)

                env.render()

                # Update observation history
                obs_vec = get_obs_vector(obs)
                obs_history.append(obs_vec)

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
    model, scheduler = load_diffusion_policy(MODEL_PATH, device=device)
    run_policy_rollout(model, scheduler, num_episodes=50, max_steps=400)

