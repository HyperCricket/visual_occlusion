import h5py, numpy as np

path = "demonstrations_20251206_203424.hdf5"

with h5py.File(path, "r") as f:
    for demo_key in list(f.keys())[:5]:
        acts = np.array(f[f"{demo_key}/actions"])  # (T, 7)
        g2A  = np.array(f[f"{demo_key}/observations/gripper_to_cubeA"])  # (T, 3)
        grq  = np.array(f[f"{demo_key}/observations/robot0_gripper_qpos"])  # (T, 2)

        dist = np.linalg.norm(g2A, axis=-1)

        # assume action[:,6] > 0.8 means "trying to close"
        close_idxs = np.where(acts[:, 6] > 0.8)[0]

        print(f"\n{demo_key}: found {len(close_idxs)} 'close' frames")
        for i in close_idxs[:10]:
            print(
                f"  i={i:4d}  action[6]={acts[i,6]:+.2f}  dist_to_A={dist[i]:.3f}  "
                f"gripper_qpos={grq[i]}"
            )

