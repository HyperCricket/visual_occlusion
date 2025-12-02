import h5py
import numpy as np
import os

PATH = "/home/kevin/Programming/Research/visual_occlusion/demonstrations_robomimic.hdf5"

if not os.path.exists(PATH):
    raise FileNotFoundError(PATH)

with h5py.File(PATH, "r+") as f:
    g_data = f["data"]
    demo_keys = sorted([k for k in g_data.keys() if k.startswith("demo_")])
    n = len(demo_keys)
    if n == 0:
        raise RuntimeError("No demo_* groups found under /data")

    # 90% train, 10% valid (at least 1 valid demo)
    split_idx = max(1, int(0.9 * n))

    train_demos = demo_keys[:split_idx]
    valid_demos = demo_keys[split_idx:]

    print(f"Found {n} demos total.")
    print(f"Train demos ({len(train_demos)}): {train_demos}")
    print(f"Valid demos ({len(valid_demos)}): {valid_demos}")

    # Remove existing mask group if present
    if "mask" in f:
        del f["mask"]

    g_mask = f.create_group("mask")

    # h5py wants bytes for string datasets ('S' dtype)
    train_arr = np.array(train_demos, dtype="S")
    valid_arr = np.array(valid_demos, dtype="S") if valid_demos else np.array([], dtype="S")

    g_mask.create_dataset("train", data=train_arr)
    g_mask.create_dataset("valid", data=valid_arr)

print("Created /mask/train and /mask/valid in", PATH)

