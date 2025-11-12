import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
from tqdm import tqdm


# ==========================================
# Dataset
# ==========================================
class RobotHDF5Dataset(Dataset):
    def __init__(self, file_path, obs_horizon=16, device="cpu"):
        self.file_path = file_path
        self.obs_horizon = obs_horizon
        self.device = device
        self.indices = []

        with h5py.File(self.file_path, "r") as f:
            for demo_key in f.keys():
                # Only load valid demo groups
                if not demo_key.startswith("demo_"):
                    continue
                if f"{demo_key}/actions" not in f:
                    continue

                T = f[f"{demo_key}/actions"].shape[0]
                for t in range(T - obs_horizon):
                    self.indices.append((demo_key, t, t + obs_horizon))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        demo_key, start, end = self.indices[idx]
        with h5py.File(self.file_path, "r") as f:
            actions = np.array(f[f"{demo_key}/actions"][start:end])  # (H, 7)

            joint_pos = np.array(f[f"{demo_key}/observations/joint_pos"][start:end])
            joint_vel = np.array(f[f"{demo_key}/observations/joint_vel"][start:end])
            gripper_qpos = np.array(f[f"{demo_key}/observations/gripper_qpos"][start:end])

            # Combine into single vector per timestep
            obs_seq = np.concatenate([joint_pos, joint_vel, gripper_qpos], axis=-1)  # (H, D)

        obs_flat = obs_seq.flatten()
        obs_tensor = torch.tensor(obs_flat, dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(actions, dtype=torch.float32, device=self.device)

        return obs_tensor, action_tensor


# ==========================================
# Diffusion Policy Model (UNet1D-like)
# ==========================================
class ConditionalUnet1D(nn.Module):
    def __init__(self, obs_dim, action_dim, horizon):
        super().__init__()
        self.input_dim = action_dim
        self.cond_dim = obs_dim * horizon

        self.net = nn.Sequential(
            nn.Linear(action_dim + self.cond_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x, obs_cond):
        inp = torch.cat([x, obs_cond], dim=-1)
        return self.net(inp)


# ==========================================
# Training function
# ==========================================
def train_diffusion_policy(hdf5_path, epochs=5, batch_size=32, obs_horizon=16, device="cpu"):
    dataset = RobotHDF5Dataset(hdf5_path, obs_horizon, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    obs_dim = 16  # joint_pos(7) + joint_vel(7) + gripper_qpos(2)
    action_dim = 7
    model = ConditionalUnet1D(obs_dim, action_dim, obs_horizon).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for obs, actions in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Flatten actions for one-step prediction
            # Could also use full sequence modeling later
            pred = model(actions[:, -1, :], obs)
            loss = criterion(pred, actions[:, -1, :])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.6f}")

    torch.save(model.state_dict(), "diffusion_policy.pth")
    print("✅ Training complete — model saved to diffusion_policy.pth")
    return model


# ==========================================
# Run everything
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_diffusion_policy(
        hdf5_path="demonstrations_20251110_162100.hdf5",
        epochs=5,
        batch_size=32,
        obs_horizon=16,
        device=device
    )
