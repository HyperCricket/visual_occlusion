#!/usr/bin/env python
"""
Utility helpers for your robomimic / robosuite HDF5 datasets.
Includes:
  - fix_env_args_type_string(path)
  - fix_env_args_type_enum(path)
  - make_train_valid_masks(path)
  - mark_train_valid(path)
  - hdfviewer_actions_per_dim(path, max_demos=5)
CLI usage examples:
  # Fix env_args["type"] to integer enum (1 = robosuite) and make masks + mark demos
  python hdf5_tools.py --mode all-robomimic --path /home/kevin/Programming/Research/visual_occlusion/demonstrations_robomimic.hdf5
  # Just view actions / distances in a raw demo file
  python hdf5_tools.py --mode view-actions --path demonstrations_20251126_225141.hdf5
"""
import os
import json
import argparse
import h5py
import numpy as np
# ---------------------------------------------------------------------------
# 1) Fix env_args["type"] to string "robosuite"
# ---------------------------------------------------------------------------
def fix_env_args_type_string(path: str):
    """
    For a robomimic-style file (with /data group), ensure env_args["type"] = "robosuite".
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with h5py.File(path, "r+") as f:
        if "data" not in f:
            raise RuntimeError("Expected /data group in file (robomimic-style file).")
        g_data = f["data"]
        if "env_args" not in g_data.attrs:
            raise RuntimeError("No 'env_args' attribute found in /data.")
        env_args_raw = g_data.attrs["env_args"]
        env_args = json.loads(env_args_raw)
        # Set type as string
        env_args["type"] = "robosuite"
        g_data.attrs["env_args"] = json.dumps(env_args)
    print("Updated env_args['type'] = 'robosuite' in", path)
# ---------------------------------------------------------------------------
# 2) Fix env_args["type"] to integer enum (1 = EnvType.ROBOSUITE_TYPE)
# ---------------------------------------------------------------------------
def fix_env_args_type_enum(path: str):
    """
    For a robomimic-style file, set env_args['type'] = 1 (EnvType.ROBOSUITE_TYPE).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with h5py.File(path, "r+") as f:
        if "data" not in f:
            raise RuntimeError("Expected /data group in file (robomimic-style file).")
        g_data = f["data"]
        if "env_args" not in g_data.attrs:
            raise RuntimeError("No 'env_args' attribute found in /data.")
        env_args_raw = g_data.attrs["env_args"]
        env_args = json.loads(env_args_raw)
        # 1 == EnvType.ROBOSUITE_TYPE
        env_args["type"] = 1
        # You can optionally override env_name if needed:
        # env_args["env_name"] = "Lift"  # or "StackWithCustomRandomization", etc.
        g_data.attrs["env_args"] = json.dumps(env_args)
    print("Updated env_args['type'] = 1 in", path)
# ---------------------------------------------------------------------------
# 3) View actions vs distance to cubeA in a raw demo file
# ---------------------------------------------------------------------------
def hdfviewer_actions_per_dim(path: str, max_demos: int = 5):
    """
    Diagnostic viewer for a raw teleop file (non-robomimic structure).
    Expects layout:
      /demo_1/actions
      /demo_1/observations/gripper_to_cubeA
      /demo_1/observations/robot0_gripper_qpos
    etc.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        demo_keys = [k for k in f.keys() if k.startswith("demo_")]
        if not demo_keys:
            print("No demo_* groups found in", path)
            return
        for demo_key in demo_keys[:max_demos]:
            acts = np.array(f[f"{demo_key}/actions"])  # (T, action_dim)
            g2A  = np.array(f[f"{demo_key}/observations/gripper_to_cubeA"])  # (T, 3)
            grq  = np.array(f[f"{demo_key}/observations/robot0_gripper_qpos"])  # (T, 2)
            dist = np.linalg.norm(g2A, axis=-1)
            # Example heuristic: action[:,6] > 0.8 means "trying to close"
            if acts.shape[1] <= 6:
                print(f"{demo_key}: action dim < 7, cannot use action[:,6] as 'close' axis.")
                continue
            close_idxs = np.where(acts[:, 6] > 0.8)[0]
            print(f"\n{demo_key}: found {len(close_idxs)} 'close' frames")
            for i in close_idxs[:10]:
                print(
                    f"  i={i:4d}  action[6]={acts[i,6]:+.2f}  dist_to_A={dist[i]:.3f}  "
                    f"gripper_qpos={grq[i]}"
                )
# ---------------------------------------------------------------------------
# 4) Create /mask/train and /mask/valid for robomimic file
# ---------------------------------------------------------------------------
def make_train_valid_masks(path: str):
    """
    For a robomimic-style file with /data/demo_* groups, create:
      /mask/train   -> list of demo names (bytes)
      /mask/valid   -> list of demo names (bytes)
    Split is 90% train, 10% valid (at least 1 valid).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with h5py.File(path, "r+") as f:
        if "data" not in f:
            raise RuntimeError("Expected /data group in file (robomimic-style file).")
        g_data = f["data"]
        demo_keys = sorted([k for k in g_data.keys() if k.startswith("demo_")])
        n = len(demo_keys)
        if n == 0:
            raise RuntimeError("No demo_* groups found under /data")
        split_idx = max(1, int(0.9 * n))
        train_demos = demo_keys[:split_idx]
        valid_demos = demo_keys[split_idx:]
        print(f"Found {n} demos total.")
        print(f"Train demos ({len(train_demos)}): {train_demos}")
        print(f"Valid demos ({len(valid_demos)}): {valid_demos}")
        # Remove existing /mask group if present
        if "mask" in f:
            del f["mask"]
        g_mask = f.create_group("mask")
        # h5py wants bytes for fixed-length string datasets
        train_arr = np.array(train_demos, dtype="S")
        valid_arr = np.array(valid_demos, dtype="S") if valid_demos else np.array([], dtype="S")
        g_mask.create_dataset("train", data=train_arr)
        g_mask.create_dataset("valid", data=valid_arr)
    print("Created /mask/train and /mask/valid in", path)
# ---------------------------------------------------------------------------
# 5) Mark each demo with train / valid attributes
# ---------------------------------------------------------------------------
def mark_train_valid(path: str):
    """
    For a robomimic-style file with /data/demo_* groups, add attrs:
      demo.attrs["train"] = 1 or 0
      demo.attrs["valid"] = 1 or 0
    using the same 90/10 split rule as make_train_valid_masks.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with h5py.File(path, "r+") as f:
        if "data" not in f:
            raise RuntimeError("Expected /data group in file (robomimic-style file).")
        g_data = f["data"]
        demo_keys = sorted([k for k in g_data.keys() if k.startswith("demo_")])
        n = len(demo_keys)
        if n == 0:
            raise RuntimeError("No demo_* groups found under /data")
        split_idx = max(1, int(0.9 * n))
        print(f"Found {n} demos. Using first {split_idx} as train, remaining {n - split_idx} as valid.")
        for i, name in enumerate(demo_keys):
            demo = g_data[name]
            is_train = 1 if i < split_idx else 0
            is_valid = 1 - is_train
            demo.attrs["train"] = is_train
            demo.attrs["valid"] = is_valid
            print(f"{name}: train={is_train}, valid={is_valid}")
    print("Done marking train/valid splits in", path)
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "fix-type-string",
            "fix-type-enum",
            "make-masks",
            "mark-splits",
            "view-actions",
            "all-robomimic",
        ],
        help="Which operation to run.",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to HDF5 file.",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=5,
        help="For view-actions: max number of demos to inspect.",
    )
    args = parser.parse_args()
    if args.mode == "fix-type-string":
        fix_env_args_type_string(args.path)
    elif args.mode == "fix-type-enum":
        fix_env_args_type_enum(args.path)
    elif args.mode == "make-masks":
        make_train_valid_masks(args.path)
    elif args.mode == "mark-splits":
        mark_train_valid(args.path)
    elif args.mode == "view-actions":
        hdfviewer_actions_per_dim(args.path, max_demos=args.max_demos)
    elif args.mode == "all-robomimic":
        # Common sequence for a converted robomimic dataset
        fix_env_args_type_enum(args.path)
        make_train_valid_masks(args.path)
        mark_train_valid(args.path)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
if __name__ == "_main_":
    main()
