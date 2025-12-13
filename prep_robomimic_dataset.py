#!/usr/bin/env python
import os, json, h5py, numpy as np
PATH = "demonstrations_robomimic_partial.hdf5"
if not os.path.exists(PATH):
    raise FileNotFoundError(PATH)
with h5py.File(PATH, "r+") as f:
    # ----- fix env_args["type"] -----
    g_data = f["data"]
    env_args_raw = g_data.attrs["env_args"]
    env_args = json.loads(env_args_raw)
    env_args["type"] = 1            # EnvType.ROBOSUITE_TYPE
    g_data.attrs["env_args"] = json.dumps(env_args)
    print("Updated env_args['type'] = 1")
    # ----- make /mask/train and /mask/valid -----
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
    if "mask" in f:
        del f["mask"]
    g_mask = f.create_group("mask")
    g_mask.create_dataset("train", data=np.array(train_demos, dtype="S"))
    g_mask.create_dataset("valid", data=np.array(valid_demos, dtype="S"))
    # ----- mark each demo with attrs -----
    for i, name in enumerate(demo_keys):
        demo = g_data[name]
        is_train = 1 if i < split_idx else 0
        is_valid = 1 - is_train
        demo.attrs["train"] = is_train
        demo.attrs["valid"] = is_valid
        print(f"{name}: train={is_train}, valid={is_valid}")
print("All done updating", PATH)
