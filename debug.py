import torch
import robomimic.utils.file_utils as FileUtils
ckpt_path = "/home/fri/visual_occlusion/experiments/diffusion_custom_demo_20251126/diffusion_custom_demo_20251126/20251206222053/last.pth"
_, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path, device="cpu", verbose=False)
print("env_name in ckpt:", ckpt_dict["env_name"])
print("env_kwargs in ckpt:", ckpt_dict["env_meta"]["env_kwargs"])
