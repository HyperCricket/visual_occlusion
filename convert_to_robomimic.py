import h5py
import json
import numpy as np
import os

IN_PATH = "/home/kevin/Programming/Research/visual_occlusion/demonstrations_20251126_225141.hdf5"
OUT_PATH = "/home/kevin/Programming/Research/visual_occlusion/demonstrations_robomimic.hdf5"

print("Reading from:", IN_PATH)
print("Writing to:", OUT_PATH)

if os.path.exists(OUT_PATH):
    raise RuntimeError(f"Output file already exists: {OUT_PATH}")

with h5py.File(IN_PATH, "r") as f_in, h5py.File(OUT_PATH, "w") as f_out:
    # Decide where demo_* groups live: root or /data
    if "data" in f_in and isinstance(f_in["data"], h5py.Group):
        base_in = f_in["data"]
        print("Detected 'data' group at root; will read demos from /data")
    else:
        base_in = f_in
        print("No 'data' group at root; will read demos from root")

    root_keys = list(base_in.keys())
    print("Top-level keys under base_in:", root_keys)

    demo_keys = [k for k in root_keys if k.startswith("demo_")]
    demo_keys = sorted(demo_keys)
    print("Found demo groups:", demo_keys)

    if not demo_keys:
        raise RuntimeError("No groups starting with 'demo_' were found. "
                           "Check your HDF5 structure.")

    # Create top-level "data" group in the NEW file
    g_data = f_out.create_group("data")

    # Minimal env_args; adjust if you need more detail for your env
    env_args = {
        "env_name": "StackWithCustomRandomization",
        "env_type": "robosuite",
        "env_kwargs": {
            "robots": "Panda",
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": False,   # set True if your demos include images
            "use_object_obs": True,
            "reward_shaping": True,
            "control_freq": 15,
        },
    }
    g_data.attrs["env_args"] = json.dumps(env_args)

    num_kept = 0

    for demo_name in demo_keys:
        demo_in = base_in[demo_name]
        in_keys = list(demo_in.keys())
        print(f"\nProcessing {demo_name}, keys: {in_keys}")

        if "actions" not in demo_in:
            print(f"  WARNING: '{demo_name}' has no 'actions' dataset. "
                  f"Skipping this demo.")
            continue

        # Create corresponding group under /data
        demo_out = g_data.create_group(demo_name)

        # actions
        actions = demo_in["actions"][...]
        demo_out.create_dataset("actions", data=actions, compression="gzip")

        # rewards (optional)
        if "rewards" in demo_in:
            rewards = demo_in["rewards"][...]
            demo_out.create_dataset("rewards", data=rewards, compression="gzip")

        # dones / terminals
        if "dones" in demo_in:
            dones = demo_in["dones"][...]
            dones_bool = np.array(dones).astype(bool)
            demo_out.create_dataset("dones", data=dones_bool, compression="gzip")
            demo_out.create_dataset("terminals", data=dones_bool, compression="gzip")

        # observations -> obs group
        if "observations" in demo_in:
            obs_in = demo_in["observations"]
            obs_out = demo_out.create_group("obs")
            for obs_key in obs_in.keys():
                data = obs_in[obs_key][...]
                obs_out.create_dataset(obs_key, data=data, compression="gzip")
        elif "obs" in demo_in:
            # already robomimic-style
            obs_in = demo_in["obs"]
            obs_out = demo_out.create_group("obs")
            for obs_key in obs_in.keys():
                data = obs_in[obs_key][...]
                obs_out.create_dataset(obs_key, data=data, compression="gzip")
        else:
            print(f"  WARNING: '{demo_name}' has no 'observations' or 'obs' group.")

        # num_samples
        num_samples = actions.shape[0]
        demo_out.attrs["num_samples"] = int(num_samples)

        # model_file attribute (we don't have a specific XML, so leave empty)
        demo_out.attrs["model_file"] = ""

        num_kept += 1
        print(f"  Kept {demo_name} with {num_samples} samples.")

    if num_kept == 0:
        raise RuntimeError("No demos with 'actions' were found / converted. "
                           "Converted file would be empty.")

print("\nConversion finished successfully.")

