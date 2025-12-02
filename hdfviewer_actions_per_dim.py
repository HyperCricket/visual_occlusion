# hdfviewer_actions_per_dim.py
import h5py, numpy as np

path = "demonstrations_20251126_225141.hdf5"
all_actions = []

with h5py.File(path, "r") as f:
    for demo_key in f.keys():
        if not demo_key.startswith("demo_"): 
            continue
        if f"{demo_key}/actions" not in f:
            continue
        acts = np.array(f[f"{demo_key}/actions"])  # (T, 7)
        all_actions.append(acts)

all_actions = np.concatenate(all_actions, axis=0)  # (N, 7)

print("Per-dim mean:", all_actions.mean(axis=0))
print("Per-dim std: ", all_actions.std(axis=0))
print("First 10 actions:\n", all_actions[:10])

