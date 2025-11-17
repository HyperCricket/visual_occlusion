import torch

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1. Load trained model + scheduler
    model, noise_scheduler = load_trained_model(device=device)

    # 2. Build an observation window from your dataset
    hdf5_path = "demonstrations_20251110_162100.hdf5"
    obs_tensor = get_last_obs_window(hdf5_path, demo_key="demo_0", obs_horizon=OBS_HORIZON)

    # 3. Sample an action
    sampled_action = sample_action_from_obs(
        model,
        noise_scheduler,
        obs_tensor,
        device=device,
        num_inference_steps=50
    )

    print("Sampled action:", sampled_action.cpu().numpy())

