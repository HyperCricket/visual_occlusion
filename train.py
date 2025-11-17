import math
import h5py
import numpy as np
from typing import Tuple, Sequence, Dict, Union, Optional, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler


# ==========================================
# Dataset
# ==========================================

class RobotHDF5Dataset(Dataset):
    """
    Dataset that reads low-dimensional robot data from an HDF5 demonstration file.

    Each sample:
      - obs: flattened sequence of observations over obs_horizon
              (joint_pos, joint_vel, gripper_qpos)
      - action: the action at the final timestep in that window (7-dim)
    """
    def __init__(self, file_path: str, obs_horizon: int = 16):
        self.file_path = file_path
        self.obs_horizon = obs_horizon
        self.indices = []  # list of (demo_key, start, end)

        with h5py.File(self.file_path, "r") as f:
            for demo_key in f.keys():
                # Only load valid demo groups
                if not demo_key.startswith("demo_"):
                    continue
                if f"{demo_key}/actions" not in f:
                    continue

                T = f[f"{demo_key}/actions"].shape[0]
                # We take windows of length obs_horizon, last action is the label
                for t in range(T - obs_horizon):
                    self.indices.append((demo_key, t, t + obs_horizon))

        # Observation dimension per timestep: joint_pos(7) + joint_vel(7) + gripper_qpos(2) = 16
        self.obs_dim = 16
        self.action_dim = 7

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        demo_key, start, end = self.indices[idx]
        with h5py.File(self.file_path, "r") as f:
            actions = np.array(f[f"{demo_key}/actions"][start:end])  # (H, 7)

            joint_pos = np.array(f[f"{demo_key}/observations/joint_pos"][start:end])      # (H, 7)
            joint_vel = np.array(f[f"{demo_key}/observations/joint_vel"][start:end])      # (H, 7)
            gripper_qpos = np.array(f[f"{demo_key}/observations/gripper_qpos"][start:end])# (H, 2)

            # Combine into single vector per timestep: (H, 16)
            obs_seq = np.concatenate([joint_pos, joint_vel, gripper_qpos], axis=-1)

        # Flatten sequence: (H, 16) -> (H*16,)
        obs_flat = obs_seq.flatten().astype(np.float32)

        # For now, we predict ONLY the last action in the window: shape (7,)
        action_last = actions[-1].astype(np.float32)

        obs_tensor = torch.from_numpy(obs_flat)       # (obs_horizon * 16,)
        action_tensor = torch.from_numpy(action_last) # (7,)

        return {
            "obs": obs_tensor,
            "action": action_tensor
        }


# ==========================================
# Conditional "UNet1D" (actually an MLP here)
# ==========================================

class ConditionalUnet1D(nn.Module):
    """
    Simple conditional network for diffusion over a single 7D action.

    Inputs:
      - x: noisy action, shape (B, action_dim)
      - t_emb: timestep embedding, shape (B, time_emb_dim)
      - obs_cond: flattened observation history, shape (B, obs_dim * horizon)

    Output:
      - predicted noise for the action, shape (B, action_dim)
    """
    def __init__(self, obs_dim: int, action_dim: int, horizon: int, time_emb_dim: int = 64):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.horizon = horizon
        self.cond_dim = obs_dim * horizon
        self.time_emb_dim = time_emb_dim

        # Simple sinusoidal time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )

        # Main MLP
        in_dim = action_dim + self.cond_dim + 128
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def timestep_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Create sinusoidal embeddings for timesteps.
        timesteps: (B,) integer tensor
        dim: embedding dimension
        """
        half = dim // 2
        # (B, 1)
        ts = timesteps.float().unsqueeze(-1)
        # frequencies
        freqs = torch.exp(
            torch.arange(half, device=timesteps.device, dtype=torch.float32)
            * -(math.log(10000.0) / (half - 1))
        )
        # (B, half)
        args = ts * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb  # (B, dim)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, obs_cond: torch.Tensor) -> torch.Tensor:
        """
        x: (B, action_dim) noisy action
        timesteps: (B,) integer steps
        obs_cond: (B, obs_dim * horizon)
        """
        # Time embedding
        t_emb_raw = self.timestep_embedding(timesteps, self.time_emb_dim)
        t_emb = self.time_embed(t_emb_raw)

        # Concatenate condition + time + noisy action
        inp = torch.cat([x, obs_cond, t_emb], dim=-1)
        return self.net(inp)


# ==========================================
# Trainer
# ==========================================

class DiffusionTrainer:
    def __init__(
        self,
        hdf5_path: str,
        obs_horizon: int = 16,
        batch_size: int = 32,
        device: str = "cpu",
        learning_rate: float = 1e-4,
        num_diffusion_iters: int = 1000,
        save_file: str = "diffusion_policy.pth"
    ):
        self.hdf5_path = hdf5_path
        self.obs_horizon = obs_horizon
        self.batch_size = batch_size
        self.device = device
        self.lr = learning_rate
        self.num_diffusion_iters = num_diffusion_iters
        self.save_file = save_file

        # From dataset definition
        self.obs_dim = 16   # per timestep
        self.action_dim = 7 # per action

    def get_dataloader(self) -> DataLoader:
        dataset = RobotHDF5Dataset(self.hdf5_path, obs_horizon=self.obs_horizon)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def get_model_and_ema(self):
        model = ConditionalUnet1D(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            horizon=self.obs_horizon
        ).to(self.device)

        ema = EMAModel(
            parameters=model.parameters(),
            power=0.75
        )
        return model, ema

    def get_noise_scheduler(self):
        return DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon"
        )

    def train(self, num_epochs: int = 20, print_stats: bool = True):
        dataloader = self.get_dataloader()
        model, ema = self.get_model_and_ema()
        noise_scheduler = self.get_noise_scheduler()

        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=self.lr,
            weight_decay=1e-6
        )

        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=len(dataloader) * num_epochs
        )

        min_loss = float("inf")
        min_epoch = 0

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                obs = batch["obs"].to(self.device)        # (B, obs_horizon * obs_dim)
                action = batch["action"].to(self.device)  # (B, action_dim)

                B = action.shape[0]

                # Sample random noise
                noise = torch.randn_like(action)

                # Sample random timesteps for each sample in the batch
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (B,),
                    device=self.device
                ).long()

                # Add noise to actions according to the diffusion schedule
                noisy_action = noise_scheduler.add_noise(
                    original_samples=action,
                    noise=noise,
                    timesteps=timesteps
                )

                # Predict the noise
                noise_pred = model(
                    x=noisy_action,
                    timesteps=timesteps,
                    obs_cond=obs
                )

                # Diffusion loss: MSE between predicted and true noise
                loss = nn.functional.mse_loss(noise_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                ema.step(model.parameters())

                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(dataloader))
            if print_stats:
                print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")

            if avg_loss < min_loss:
                min_loss = avg_loss
                min_epoch = epoch + 1

        # Save final model parameters (EMA-averaged)
        ema.copy_to(model.parameters())
        torch.save(model.state_dict(), self.save_file)
        print(f"\n Training complete — EMA model saved to {self.save_file}")
        print(f"   Minimum epoch loss: {min_loss:.6f} at epoch {min_epoch}")

        return model


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    trainer = DiffusionTrainer(
        hdf5_path="demonstrations_20251110_162100.hdf5",
        obs_horizon=16,
        batch_size=32,
        device=device,
        learning_rate=1e-4,
        num_diffusion_iters=1000,
        save_file="diffusion_policy_robot.pth"
    )

    model = trainer.train(num_epochs=20, print_stats=True)

