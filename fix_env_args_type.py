import h5py
import json

PATH = "/home/kevin/Programming/Research/visual_occlusion/demonstrations_robomimic.hdf5"

with h5py.File(PATH, "r+") as f:
    g_data = f["data"]
    env_args_raw = g_data.attrs["env_args"]
    env_args = json.loads(env_args_raw)

    # Add or fix the 'type' field
    env_args["type"] = "robosuite"

    # Optional: you can keep or drop 'env_type'; robomimic doesn't care
    # env_args.pop("env_type", None)

    g_data.attrs["env_args"] = json.dumps(env_args)

print("Updated env_args in", PATH)

