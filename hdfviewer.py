import h5py

with h5py.File("demonstrations_20251110_162100.hdf5", "r") as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"DATASET: {name} -> shape={obj.shape}")
        else:
            print(f"GROUP: {name}")
    f.visititems(print_structure)
