#!/usr/bin/env python

import os
import argparse
import numpy as np
import torch
import imageio

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.algo.algo import RolloutPolicy
from robosuite.environments.base import register_env
from control import StackWithCustomRandomization


register_env(StackWithCustomRandomization)

# -------------------------
# Occlusion utilities
# -------------------------

def apply_center_occlusion(obs, image_keys, frac=0.33):
    """
    Very simple visual occlusion: zero out a central square in each specified image.

    obs: dict from env.reset() / env.step()
    image_keys: list of keys in obs that are image observations (e.g. 'agentview_image')
    frac: fraction of min(H, W) used as the occluded square size
    """
    if obs is None:
        return obs

    # Shallow copy dict so we don't mutate env's internal obs
    new_obs = dict(obs)

    for k in image_keys:
        if k not in new_obs:
            continue
        img = new_obs[k]
        # Expect shape (H, W, C) or (C, H, W). Robomimic usually uses (C, H, W).
        arr = np.array(img)

        if arr.ndim != 3:
            continue

        # handle channel-first vs channel-last
        channel_first = arr.shape[0] in (1, 3)
        if channel_first:
            # (C, H, W) -> (H, W, C) for easy masking
            arr_hw = np.moveaxis(arr, 0, -1)
        else:
            arr_hw = arr

        h, w, _ = arr_hw.shape
        side = int(min(h, w) * frac)
        cy, cx = h // 2, w // 2
        y0 = max(cy - side // 2, 0)
        y1 = min(cy + side // 2, h)
        x0 = max(cx - side // 2, 0)
        x1 = min(cx + side // 2, w)

        arr_hw[y0:y1, x0:x1, :] = 0

        if channel_first:
            arr = np.moveaxis(arr_hw, -1, 0)
        else:
            arr = arr_hw

        new_obs[k] = arr

    return new_obs


def apply_occlusion(obs, mode, image_keys):
    """
    Wrapper to choose occlusion mode.
    Extend this with your own occlusion functions later.
    """
    if mode == "none":
        return obs
    elif mode == "center":
        return apply_center_occlusion(obs, image_keys)
    else:
        raise ValueError(f"Unknown occlusion mode: {mode}")


# -------------------------
# Rollout loop
# -------------------------

def rollout(
    policy,
    env,
    horizon,
    render=False,
    video_writer=None,
    video_skip=5,
    camera_names=None,
    occlusion_mode="none",
    image_obs_keys=None,
):
    """
    Run one episode with a trained robomimic RolloutPolicy.

    policy: RolloutPolicy from FileUtils.policy_from_checkpoint
    env: EnvBase from FileUtils.env_from_checkpoint
    horizon: max steps
    render: if True, on-screen rendering
    video_writer: imageio writer or None
    video_skip: write every N-th frame to video
    camera_names: list of camera names for rendering
    occlusion_mode: 'none' or 'center' (you can add more)
    image_obs_keys: which obs keys correspond to images for occlusion
    """
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    if camera_names is None:
        # Robosuite Lift default cameras
        camera_names = ["agentview", "robot0_eye_in_hand"]

    if image_obs_keys is None:
        # In robomimic image datasets, obs keys are usually like "<camera>_image"
        image_obs_keys = [f"{cam}_image" for cam in camera_names]

    stats = {
        "return": 0.0,
        "length": 0,
        "success": False,
    }
    traj = {
        "obs": [],
        "actions": [],
        "rewards": [],
    }

    # Prepare policy for a new episode
    policy.start_episode()
    obs = env.reset()

    for t in range(horizon):
        # Apply occlusion BEFORE feeding to policy
        obs_for_policy = apply_occlusion(obs, occlusion_mode, image_obs_keys)

        # Get action from robomimic rollout policy
        action = policy(obs_for_policy)

        next_obs, reward, done, info = env.step(action)

        stats["return"] += float(reward)
        stats["length"] += 1

        success = False
        if hasattr(env, "is_success"):
            success = bool(env.is_success().get("task", False))
        stats["success"] = success

        traj["obs"].append(obs)
        traj["actions"].append(action)
        traj["rewards"].append(float(reward))

        if render:
            env.render()

        if video_writer is not None:
            frames = []
            for cam in camera_names:
                frame = env.render(
                    mode="rgb_array",
                    camera_name=cam,
                    height=256,
                    width=256,
                )
                frames.append(frame)

        if len(frames) > 0 and (t % video_skip == 0):
            frame = np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]
            video_writer.append_data(frame)


        if done or success:
            break

        obs = next_obs

    return stats, traj


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to robomimic checkpoint (.pth), e.g. last.pth",
    )
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=10,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Max steps per episode (defaults to config.experiment.rollout.horizon)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable on-screen rendering",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="If set, write a video (per-episode file with _ep#.mp4 suffix)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for NumPy / Torch",
    )
    parser.add_argument(
        "--occlusion",
        type=str,
        default="none",
        choices=["none", "center"],
        help="Occlusion mode to apply to image observations",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Try to run policy on CUDA if available",
    )

    args = parser.parse_args()

    # Device selection
    device = TorchUtils.get_torch_device(try_to_use_cuda=args.use_gpu)
    print(f"Using device: {device}")

    # Load trained policy and checkpoint dict
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.ckpt_path,
        device=device,
        verbose=True,
    )

    # Get config & rollout horizon
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    horizon = args.horizon or config.experiment.rollout.horizon
    print(f"Using rollout horizon = {horizon}")

    # Build environment from checkpoint metadata
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=None,                 # use env that policy was trained on
        render=args.render,
        render_offscreen=(args.video_path is not None),
        verbose=True,
    )

    # Seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Decide camera + obs keys (for Lift, this is correct)
    camera_names = ["agentview", "robot0_eye_in_hand"]
    image_obs_keys = [f"{cam}_image" for cam in camera_names]

    # Optional: video
    video_base = args.video_path
    if video_base is not None and not video_base.endswith(".mp4"):
        video_base = video_base + ".mp4"

    all_returns = []
    num_success = 0

    for ep in range(args.n_rollouts):
        if video_base is not None:
            import imageio
            video_path_ep = video_base.replace(".mp4", f"_ep{ep}.mp4")
            video_writer = imageio.get_writer(video_path_ep, fps=20)
        else:
            video_writer = None

        stats, traj = rollout(
            policy=policy,
            env=env,
            horizon=horizon,
            render=args.render,
            video_writer=video_writer,
            video_skip=5,
            camera_names=camera_names,
            occlusion_mode=args.occlusion,
            image_obs_keys=image_obs_keys,
        )

        if video_writer is not None:
            video_writer.close()

        all_returns.append(stats["return"])
        num_success += int(stats["success"])

        print(
            f"[Episode {ep}] return={stats['return']:.3f}, "
            f"len={stats['length']}, success={stats['success']}"
        )

    avg_return = float(np.mean(all_returns)) if len(all_returns) > 0 else 0.0
    success_rate = num_success / float(max(1, args.n_rollouts))

    print("========================================")
    print(f"Average return over {args.n_rollouts} episodes: {avg_return:.3f}")
    print(f"Success rate: {success_rate * 100:.1f}%")
    print("========================================")


if __name__ == "__main__":
    main()

