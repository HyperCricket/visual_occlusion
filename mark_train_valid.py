import h5py
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

    print(f"Found {n} demos. Using first {split_idx} as train, remaining {n - split_idx} as valid.")

    for i, name in enumerate(demo_keys):
        demo = g_data[name]
        is_train = 1 if i < split_idx else 0
        is_valid = 1 - is_train

        demo.attrs["train"] = is_train
        demo.attrs["valid"] = is_valid

        print(f"{name}: train={is_train}, valid={is_valid}")

print("Done marking train/valid splits.")

