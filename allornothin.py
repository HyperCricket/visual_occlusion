# ======================================================
# 🧩 Diffusion Policy Training Pipeline for HDF5 Demos
# ======================================================

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import h5py
import random
from tqdm.auto import tqdm
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
import torchvision

# ======================================================
# 1️⃣ Dataset
# ======================================================

class RobotHDF5Dataset(Dataset):
    def __init__(self, hdf5_path, obs_horizon=16, device="cpu"):
        self.hdf5_path = hdf5_path
        self.obs_horizon = obs_horizon
        self.device = device

        self.demos = []
        with h5py.File(hdf5_path, "r") as f:
            for key in f.keys():
                length = f[f"{key}/action"].shape[0]
                if length >= obs_horizon:
                    self.demos.append((key, length))

    def __len__(self):
        return len(self.demos) * 10  # more sampling flexibility

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, "r") as f:
            demo_key, demo_len = random.choice(self.demos)
            start = random.randint(0, demo_len - self.obs_horizon)
            end = start + self.obs_horizon

            obs_img = np.array(f[f"{demo_key}/obs/image"][start:end])
            obs_low = np.array(f[f"{demo_key}/obs/low_dim"][start:end])
            action = np.array(f[f"{demo_key}/action"][start:end])

        obs_img = torch.from_numpy(obs_img).permute(0, 3, 1, 2).float() / 255.0
        obs_low = torch.from_numpy(obs_low).float()
        action = torch.from_numpy(action).float()

        return obs_img.to(self.device), obs_low.to(self.device), action.to(self.device)

# ======================================================
# 2️⃣ Network Components
# ======================================================

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
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale, bias = embed[:, 0, ...], embed[:, 1, ...]
        out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

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

class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, diffusion_step_embed_dim=256,
                 down_dims=[256, 512, 1024], kernel_size=5, n_groups=8):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]

        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim),
        ])

        self.down_modules = nn.ModuleList([
            nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim),
                Downsample1d(dim_out) if i < len(in_out)-1 else nn.Identity()
            ])
            for i, (dim_in, dim_out) in enumerate(in_out)
        ])

        self.up_modules = nn.ModuleList([
            nn.ModuleList([
                ConditionalResidualBlock1D(dim_out*2, dim_in, cond_dim),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim),
                Upsample1d(dim_in) if i < len(in_out)-1 else nn.Identity()
            ])
            for i, (dim_in, dim_out) in enumerate(reversed(in_out[1:]))
        ])

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self, sample, timestep, global_cond):
        sample = sample.moveaxis(-1, -2)  # (B, C, T)
        timesteps = timestep.expand(sample.shape[0])
        global_feature = self.diffusion_step_encoder(timesteps)
        global_feature = torch.cat([global_feature, global_cond], dim=-1)

        x, skips = sample, []
        for res1, res2, down in self.down_modules:
            x = res1(x, global_feature)
            x = res2(x, global_feature)
            skips.append(x)
            x = down(x)

        for mid in self.mid_modules:
            x = mid(x, global_feature)

        for res1, res2, up in self.up_modules:
            x = torch.cat((x, skips.pop()), dim=1)
            x = res1(x, global_feature)
            x = res2(x, global_feature)
            x = up(x)

        x = self.final_conv(x)
        return x.moveaxis(-1, -2)

# ======================================================
# 3️⃣ Vision Encoder
# ======================================================

def get_resnet(name="resnet18", weights="IMAGENET1K_V1"):
    resnet = getattr(torchvision.models, name)(weights=weights)
    resnet.fc = nn.Identity()
    return resnet

def replace_bn_with_gn(model, groups=16):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features
            setattr(model, name, nn.GroupNorm(num_channels // groups, num_channels))
        else:
            replace_bn_with_gn(module, groups)
    return model

# ======================================================
# 4️⃣ Training Setup
# ======================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = RobotHDF5Dataset("demonstrations.hdf5", obs_horizon=16, device=device)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

vision_encoder = get_resnet("resnet18").to(device)
vision_encoder = replace_bn_with_gn(vision_encoder)
vision_feature_dim = 512
lowdim_obs_dim = 7
action_dim = 7
obs_horizon = 16
obs_dim = vision_feature_dim + lowdim_obs_dim

noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim * obs_horizon
).to(device)

ema = EMAModel(parameters=noise_pred_net.parameters(), power=0.75)
optimizer = torch.optim.AdamW(noise_pred_net.parameters(), lr=1e-4)
scheduler = DDPMScheduler(num_train_timesteps=1000)

# ======================================================
# 5️⃣ Training Loop
# ======================================================

EPOCHS = 10

for epoch in range(EPOCHS):
    noise_pred_net.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for obs_img, obs_low, actions in pbar:
        B, T, _, _, _ = obs_img.shape

        # Encode visual observations
        obs_img_flat = obs_img.reshape(B*T, 3, obs_img.shape[-2], obs_img.shape[-1])
        vision_feats = vision_encoder(obs_img_flat)
        vision_feats = vision_feats.reshape(B, T, -1)

        # Combine obs
        obs_combined = torch.cat([vision_feats, obs_low], dim=-1)
        obs_cond = obs_combined.reshape(B, -1)

        # Diffusion step and noise
        t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device).long()
        noise = torch.randn_like(actions)
        noisy_actions = scheduler.add_noise(actions, noise, t)

        # Predict noise
        pred_noise = noise_pred_net(noisy_actions, t, global_cond=obs_cond)

        # Loss
        loss = F.mse_loss(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.step(noise_pred_net.parameters())

        pbar.set_postfix(loss=loss.item())

    torch.save({
        "vision_encoder": vision_encoder.state_dict(),
        "noise_pred_net": noise_pred_net.state_dict(),
        "ema": ema.state_dict(),
    }, f"checkpoint_epoch_{epoch+1}.pth")

print("✅ Training complete!")
