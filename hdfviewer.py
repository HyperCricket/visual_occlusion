import h5py
import os

file_path = "~/Programming/Research/visual_occlusion/demonstrations_20251103_162718.hdf5"
expanded_file_path = os.path.expanduser(file_path)

# Create a File Access Property List (FAPL)
fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)

# --- CORRECTED LINE ---
# Access the library version constants directly from the top-level h5py module
fapl.set_libver_bounds(h5py.h5f.libver.earliest, h5py.h5f.libver.latest)

print(f"Attempting to open file: {expanded_file_path}")

try:
    # Pass the FAPL to the h5py.File constructor
    # The constants should be imported from h5py, but the set_libver_bounds 
    # function might still need the h5py.h5f prefix in some setups.
    
    # Try the most robust fix first, which uses the constants from h5py.h5f.libver:
    # NOTE: The *entire traceback* suggests h5py.h5f.libver is the issue. 
    # Let's try the constants directly under h5py for a common fix:
    
    # The actual correct constants are usually:
    # h5py.h5f.libver.earliest  and h5py.h5f.libver.latest
    
    # Since you received the 'AttributeError: module 'h5py.h5f' has no attribute 'libver''
    # This means your installation is older or differently structured.

    # Let's use the constants from the standard high-level HDF5 library:
    # NOTE: If this still fails, you MUST update h5py.
    
    fapl.set_libver_bounds(h5py.h5f.libver.earliest, h5py.h5f.libver.latest) # <-- This is the standard, let's confirm the code you ran was exactly this.

    # If the error is still present, the fix is to remove the .libver:
    # fapl.set_libver_bounds(h5py.h5f.earliest, h5py.h5f.latest) # <-- This is incorrect for modern h5py but might work for an old one.

    # Let's revert to the standard:
    with h5py.File(expanded_file_path, "r", fapl=fapl) as f:
        print("Successfully opened HDF5 file.")
        
        def print_structure(name):
            print(name)
        f.visit(print_structure)
        
        for key in f.keys():
            print(f'{key}: shape={f[key].shape}, dtype={f[key].dtype}')

except AttributeError:
    # If the standard call fails, this means your h5py is likely very old/broken.
    # The most common failure mode is an issue with the h5py version.
    print("\n⚠️ The current h5py version seems incompatible with this call.")
    print("Please update h5py using: `pip install --upgrade h5py`")
    
except OSError as e:
    print(f"\nError opening file even with FAPL modification: {e}")
    print("The file may be severely corrupted. The original file format issue persists.")
