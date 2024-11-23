import h5py
import zarr
import os

def hdf5_to_zarr_simple(hdf5_path, zarr_path):
    """
    Convert a simple HDF5 file with one dataset to a Zarr file.

    Parameters:
    - hdf5_path (str): Path to the input HDF5 file.
    - zarr_path (str): Path to the output Zarr file.
    """
    with h5py.File(hdf5_path, "r") as h5f:
        # Get the name of the only dataset
        dataset_name = next(iter(h5f.keys()))
        
        # Access the dataset
        dataset = h5f[dataset_name][:]

        print(f"Dataset name: {dataset_name}")
        print(f"Shape: {dataset.shape}")
        #print(f"Chunks: {dataset.chunks}")
        print(f"Size: {dataset.size}")

        # Convert to Zarr
        zarr.save(zarr_path, dataset[()])

    print(f"Converted HDF5 file '{hdf5_path}' to Zarr file '{zarr_path}'")


zarr_path = os.path.join(os.getcwd(), "data", "spikes-N1250-T10e06.zarr")
hdf5_path = os.path.join(os.getcwd(), "data", "spikes-N1250-T10e06.h5")

hdf5_to_zarr_simple(hdf5_path, zarr_path)

