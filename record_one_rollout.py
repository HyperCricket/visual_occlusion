#!/usr/bin/env python

import argparse
import os
import numpy as np
import torch
import imageio

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.env_utils as EnvUtils

# import + register your custom env
from robosuite.environments.base import register_env
from control import StackWithCustomRandomization

# register the class with robosuite so robosuite.make(...) can find it
register_env(StackWithCustomRandomization)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to robomimic checkpoint (.pth), e.g. last.pth",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="Output video path (e.g. /home/fri/visual_occlusion/one_rollout.mp4)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Max steps per episode (defaults to config.experiment.rollout.horizon)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Try to run policy on CUDA if available",
    )
    args = parser.parse_args()

    device = TorchUtils.get_torch_device(try_to_use_cuda=args.use_gpu)
    print(f"Using device: {device}")

    # Load policy + checkpoint
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.ckpt_path,
        device=device,
        verbose=True,
    )
    policy.eval()

    # Get config + horizon
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    horizon = args.horizon or config.experiment.rollout.horizon
    print(f"Using rollout horizon = {horizon}")

    # Build env with offscreen rendering so env.render(mode='rgb_array') works
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=None,
        render=False,
        render_offscreen=True,
        verbose=True,
    )

    # Choose cameras – adjust if needed
    camera_names = ["agentview", "robot0_eye_in_hand"]

    # Run ONE episode and save frames
    obs = env.reset()
    policy.start_episode()

    frames = []

    for t in range(horizon):
        # policy takes raw obs dict
        action = policy(obs)
        obs, reward, done, info = env.step(action)

        # grab frames from cameras
        cam_frames = []
        for cam in camera_names:
            frame = env.render(
                mode="rgb_array",
                camera_name=cam,
            )
            cam_frames.append(frame)

        if len(cam_frames) == 1:
            frame = cam_frames[0]
        else:
            frame = np.concatenate(cam_frames, axis=1)

        frames.append(frame)

        if done:
            break

    os.makedirs(os.path.dirname(args.video_path), exist_ok=True)
    imageio.mimsave(args.video_path, frames, fps=20)
    print(f"Saved rollout video with {len(frames)} frames to {args.video_path}")


if __name__ == "__main__":
    main()
