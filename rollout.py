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
# Model definition (must match training)
# ============================================================

# class ConditionalUnet1D(nn.Module):
    # """
    # Simple conditional network for diffusion over a single 7D action.
# 
    # Inputs:
      # - x: noisy action, shape (B, action_dim)
      # - timesteps: (B,) integer tensor
      # - obs_cond: flattened observation history, shape (B, obs_dim * horizon)
# 
    # Output:
      # - predicted noise for the action, shape (B, action_dim)
    # """
    # def __init__(self, obs_dim: int, action_dim: int, horizon: int, time_emb_dim: int = 64):
        # super().__init__()
        # self.action_dim = action_dim
        # self.obs_dim = obs_dim
        # self.horizon = horizon
        # self.cond_dim = obs_dim * horizon
        # self.time_emb_dim = time_emb_dim
# 
        # # Time embedding MLP
        # self.time_embed = nn.Sequential(
            # nn.Linear(time_emb_dim, 128),
            # nn.SiLU(),
            # nn.Linear(128, 128),
            # nn.SiLU(),
        # )
# 
        # # Main MLP
        # in_dim = action_dim + self.cond_dim + 128
        # self.net = nn.Sequential(
            # nn.Linear(in_dim, 256),
            # nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Linear(256, action_dim)
        # )
# 
    # def timestep_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        # """
        # Create sinusoidal embeddings for timesteps.
        # timesteps: (B,) integer tensor
        # dim: embedding dimension
        # """
        # half = dim // 2
        # ts = timesteps.float().unsqueeze(-1)  # (B, 1)
        # freqs = torch.exp(
            # torch.arange(half, device=timesteps.device, dtype=torch.float32)
            # * -(math.log(10000.0) / (half - 1))
        # )  # (half,)
        # args = ts * freqs  # (B, half)
        # emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, 2*half)
        # if dim % 2 == 1:
            # emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        # return emb  # (B, dim)
# 
    # def forward(self, x: torch.Tensor, timesteps: torch.Tensor, obs_cond: torch.Tensor) -> torch.Tensor:
        # """
        # x: (B, action_dim)    - noisy action
        # timesteps: (B,)       - integer diffusion steps
        # obs_cond: (B, obs_dim * horizon)
        # """
        # t_emb_raw = self.timestep_embedding(timesteps, self.time_emb_dim)   # (B, time_emb_dim)
        # t_emb = self.time_mlp(t_emb_raw)                                    # (B, 128)
        # inp = torch.cat([x, obs_cond, t_emb], dim=-1)                       # (B, action_dim + cond_dim + 128)
        # return self.net(inp)                                                # (B, action_dim)
 
# ============================================================
# Observation helper (low-dim, matches our training)
# ============================================================

def get_obs_vector(env, robot, arm_name="right"):
def get_obs_vector(obs):
    """
    Build the per-timestep observation vector from the env obs dict.
    MUST match what train.py concatenates, in the same order.
    """
    return np.concatenate([
        obs["robot0_joint_pos"],        # (7,)
        obs["robot0_joint_vel"],        # (7,)
        obs["robot0_gripper_qpos"],     # (2,)

        obs["cubeA_pos"],               # (3,)
        obs["cubeB_pos"],               # (3,)
        obs["gripper_to_cubeA"],        # (3,)
        obs["gripper_to_cubeB"],        # (3,)
    ]).astype(np.float32)
 
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

OBS_DIM = 16           # per timestep: 7+7+2
ACTION_DIM = 7
OBS_HORIZON = 16
NUM_DIFFUSION_ITERS = 1000
MODEL_PATH = "diffusion_policy_robot.pth"   # change if you used a different filename

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
            # neutral_joints = np.array([0, -0.3, 0, -2.0, 0, 1.7, 0.785])
            # noise = np.random.uniform(-0.2, 0.2, size=7)
            # robot.set_robot_joint_positions(neutral_joints + noise)
            # env.sim.forward()

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
                obs, reward, done, info = env.step(action_np)
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

