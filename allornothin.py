# ============================================================
#  Imports
# ============================================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import h5py
from tqdm.auto import tqdm

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

# ============================================================
#  Dataset
# ============================================================
class RobotHDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, obs_horizon=16, device="cpu"):
        self.file_path = file_path
        self.obs_horizon = obs_horizon
        self.device = device
        self.indices = []

        with h5py.File(self.file_path, "r") as f:
            for demo_key in f.keys():
                if not demo_key.startswith("demo_"):
                    continue
                T = f[f"{demo_key}/actions"].shape[0]
                for t in range(T - obs_horizon):
                    self.indices.append((demo_key, t, t + obs_horizon))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        demo_key, start, end = self.indices[idx]
        with h5py.File(self.file_path, "r") as f:
            actions = np.array(f[f"{demo_key}/actions"][start:end])

            joint_pos = np.array(f[f"{demo_key}/observations/joint_pos"][start:end])
            joint_vel = np.array(f[f"{demo_key}/observations/joint_vel"][start:end])
            gripper_qpos = np.array(f[f"{demo_key}/observations/gripper_qpos"][start:end])

            low_dim = np.concatenate([joint_pos, joint_vel, gripper_qpos], axis=-1)

        obs_flat = low_dim.flatten()
        obs_tensor = torch.tensor(obs_flat, dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(actions, dtype=torch.float32, device=self.device)

        return obs_tensor, action_tensor


# ============================================================
#  Network components
# ============================================================
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
        ])

        cond_channels = out_channels * 2
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

        self.out_channels = out_channels

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale, bias = embed[:, 0], embed[:, 1]
        out = scale * out + bias
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed)
        )

        cond_dim = dsed + global_cond_dim

        # Down
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        self.down_modules = nn.ModuleList([
            nn.ModuleList([
                ConditionalResidualBlock1D(inc, outc, cond_dim),
                ConditionalResidualBlock1D(outc, outc, cond_dim),
                Downsample1d(outc) if i < len(in_out) - 1 else nn.Identity()
            ])
            for i, (inc, outc) in enumerate(in_out)
        ])

        # Mid
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim)
        ])

        # Up
        self.up_modules = nn.ModuleList([
            nn.ModuleList([
                ConditionalResidualBlock1D(outc*2, inc, cond_dim),
                ConditionalResidualBlock1D(inc, inc, cond_dim),
                Upsample1d(inc) if i < len(in_out) - 1 else nn.Identity()
            ])
            for i, (inc, outc) in enumerate(reversed(in_out[1:]))
        ])

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size),
            nn.Conv1d(start_dim, input_dim, 1)
        )

    def forward(self, sample, timestep, global_cond):
        x = sample.moveaxis(-1, -2)  # (B, C, T)
        timesteps = timestep.expand(x.shape[0])
        global_feature = self.diffusion_step_encoder(timesteps)
        global_feature = torch.cat([global_feature, global_cond], dim=-1)

        h = []
        for res1, res2, down in self.down_modules:
            x = res1(x, global_feature)
            x = res2(x, global_feature)
            h.append(x)
            x = down(x)

        for mid in self.mid_modules:
            x = mid(x, global_feature)

        for res1, res2, up in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = res1(x, global_feature)
            x = res2(x, global_feature)
            x = up(x)

        x = self.final_conv(x)
        return x.moveaxis(-1, -2)


# ============================================================
#  Training Loop
# ============================================================
def train_diffusion_policy(hdf5_path, epochs=10, batch_size=8, obs_horizon=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RobotHDF5Dataset(hdf5_path, obs_horizon, device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    obs_dim = 7 + 7 + 2  # joint pos + vel + gripper
    action_dim = 7
    model = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_horizon * obs_dim
    ).to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    ema = EMAModel(parameters=model.parameters(), power=0.75)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for obs, actions in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()

            # sample noise
            noise = torch.randn_like(actions)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (actions.shape[0],), device=device, dtype=torch.long
            )
            noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)

            # predict noise
            pred_noise = model(noisy_actions, timesteps, obs)

            loss = nn.functional.mse_loss(pred_noise, noise)
            loss.backward()
            optimizer.step()
            ema.step(model.parameters())
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} | Avg loss: {epoch_loss/len(loader):.6f}")

    return model, ema


# ============================================================
#  Run Training
# ============================================================
if __name__ == "__main__":
    model, ema = train_diffusion_policy(
        "demonstrations_20251110_162100.hdf5",
        epochs=20,
        batch_size=8,
        obs_horizon=16
    )

    torch.save(model.state_dict(), "diffusion_policy_trained.pth")
    print("✅ Training complete, model saved as diffusion_policy_trained.pth")
