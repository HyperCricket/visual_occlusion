import h5py
import numpy as np

path = "demonstrations_20251124_211825.hdf5"  # change to your NEW file name
with h5py.File(path, "r") as f:
    print("Top-level keys:", list(f.keys())[:5])
    demo0 = [k for k in f.keys() if k.startswith("demo_")][0]
    print("Example demo:", demo0)
    print("Demo keys:", list(f[demo0].keys()))
    if "observations" in f[demo0]:
        print("Observation keys:", list(f[f"{demo0}/observations"].keys()))
        for k in f[f"{demo0}/observations"].keys():
            print(k, f[f"{demo0}/observations/{k}"].shape)
    print("Actions shape:", f[f"{demo0}/actions"].shape)

