# hdfviewer.py
import h5py
import numpy as np

path = "demonstrations_20251126_225141.hdf5"  # adjust if needed

all_actions = []

with h5py.File(path, "r") as f:
    print("Top-level keys:", list(f.keys()))
    for demo_key in f.keys():
        # Only process real demos, same as RobotHDF5Dataset
        if not demo_key.startswith("demo_"):
            print(f"Skipping non-demo group: {demo_key}")
            continue

        if f"{demo_key}/actions" not in f:
            print(f"Skipping {demo_key}: no 'actions' dataset")
            continue

        acts = np.array(f[f"{demo_key}/actions"])  # (T, 7)
        print(f"{demo_key}: actions shape {acts.shape}")
        all_actions.append(acts)

if not all_actions:
    print("No demos with actions found.")
else:
    all_actions = np.concatenate(all_actions, axis=0)  # (N, 7)
    norms = np.linalg.norm(all_actions, axis=-1)

    print("\n=== Action stats across all demos ===")
    print("actions shape:", all_actions.shape)
    print("action norm mean:", norms.mean())
    print("action norm std:", norms.std())
    print("action norm max:", norms.max())
    print("first few actions:\n", all_actions[:5])

