import h5py

file_path = "your_file.h5"

with h5py.File(file_path, "r") as f:
    # Print all the groups and datasets
    def print_structure(name):
        print(name)
    f.visit(print_structure)

    # Optionally check for empty datasets
    for key in f.keys():
        print(f"{key}: shape={f[key].shape}, dtype={f[key].dtype}")
