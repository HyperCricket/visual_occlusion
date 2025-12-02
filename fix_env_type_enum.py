import h5py
import json

PATH = "/home/kevin/Programming/Research/visual_occlusion/demonstrations_robomimic.hdf5"

with h5py.File(PATH, "r+") as f:
    g_data = f["data"]
    env_args_raw = g_data.attrs["env_args"]
    env_args = json.loads(env_args_raw)

    # Set type to EnvType.ROBOSUITE_TYPE = 1
    env_args["type"] = 1

    # (Optional) you can leave env_name as-is or just use a dummy like "Lift"
    # env_args["env_name"] = "Lift"

    g_data.attrs["env_args"] = json.dumps(env_args)

print("Updated env_args['type']=1 in", PATH)

