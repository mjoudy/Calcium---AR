import os
import sys
import zarr
import json
import numpy as np
import dask.array as da
from dask_jobqueue.slurm import SLURMCluster
from dask.distributed import Client
from functions import calsim as cs
from functions import utils as ut

'''
def main(spikes_path, tau):
    
    dask_spikes = da.from_zarr(spikes_path)
    print(dask_spikes)
    print(f"Shape: {dask_spikes.shape}")
    print(f"Chunks: {dask_spikes.chunks}")
    print(f"Data type: {dask_spikes.dtype}")

    calcium = dask_spikes.map_blocks(cs.sim_calcium, tau, dtype=np.float64)

    output_name = ut.nameit("calcium", tau=tau)
    output_zarr_file = os.path.join(os.getcwd(), "data", output_name)
    print(output_zarr_file)

    calcium.to_zarr(output_zarr_file, overwrite=True )
    print(f"Processed data saved to {output_zarr_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Value of tau should be passed!")
        sys.exit(1)

    tau = float(sys.argv[1])

    # Read the spike file name
    with open("temp_names.json", "r") as file:
        file_names = json.load(file)
        sp_name = file_names["sp_file"]

    spikes_path = os.path.join(os.getcwd(), "data", sp_name)
    print(f"Spikes address: {spikes_path}")

    cluster = SLURMCluster(
        queue="dev_single",  # Replace with your Slurm partition
        cores=1,                      # 1 CPU per worker
        memory="8GB",                 # Memory per worker
        processes=1,                  # 1 process per task
        walltime="00:30:00",          # Adjust as needed
        job_extra_directives=["--ntasks=1"],     # Ensure one task per worker
        log_directory="./logs",       # Logs directory
    )
    cluster.scale(jobs=20)  # Scale to match the number of tasks (40 workers)

    # Connect the Dask client
    client = Client(cluster)
    print(cluster)
    print(client)

    # Run the main function
    main(spikes_path, tau)

    client.close()
'''

def main(spikes_path, tau):
    """
    Main processing function to simulate calcium signals and save the output.
    """
    dask_spikes = da.from_zarr(spikes_path)
    print(dask_spikes)
    print(f"Shape: {dask_spikes.shape}")
    print(f"Chunks: {dask_spikes.chunks}")
    print(f"Data type: {dask_spikes.dtype}")

    calcium = dask_spikes.map_blocks(cs.sim_calcium, tau, dtype=np.float64)

    input_file_name = os.path.basename(spikes_path)
    output_name = ut.nameit(input_file_name, "calcium", tau=tau)
    output_zarr_file = os.path.join(os.getcwd(), "data", output_name)
    print(output_zarr_file)

    calcium.to_zarr(output_zarr_file, overwrite=True)
    print(f"Processed data saved to {output_zarr_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <spikes_path> <tau>")
        sys.exit(1)

    spikes_path = sys.argv[1]
    tau = float(sys.argv[2])

    print(f"Spikes address: {spikes_path}")

    cluster = SLURMCluster(
        queue="dev_single",  # Replace with your Slurm partition
        cores=1,                      # 1 CPU per worker
        memory="8GB",                 # Memory per worker
        processes=1,                  # 1 process per task
        walltime="00:30:00",          # Adjust as needed
        job_extra_directives=["--ntasks=1"],     # Ensure one task per worker
        log_directory="./logs",       # Logs directory
    )
    cluster.scale(jobs=20)  # Scale to match the number of tasks (40 workers)

    # Connect the Dask client
    client = Client(cluster)
    print(cluster)
    print(client)

    # Run the main function
    main(spikes_path, tau)

    client.close()