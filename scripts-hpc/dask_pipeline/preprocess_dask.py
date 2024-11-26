import os
import sys
import zarr
import json
import numpy as np
import dask.array as da
from dask_jobqueue.slurm import SLURMCluster
from dask.distributed import Client
from functions import dask_functions as df
from functions import utils as ut

def main(calcium_signal, spikes_trains, win_len, sg_win):
    """
    Main function to preprocess the calcium signal.
    """
    dask_calcium = da.from_zarr(calcium_signal)
    print(dask_calcium)
    print(f"Shape: {dask_calcium.shape}")
    print(f"Chunks: {dask_calcium.chunks}")
    print(f"Data type: {dask_calcium.dtype}")
    
    dask_spikes = da.from_zarr(spikes_trains)
    preprocessed_feed = dask_calcium.map_blocks(df.dask_feed_raper, dask_spikes, win_len, sg_win, dtype=np.float64)
    slopes = dask_calcium.map_blocks(df.dask_fits_raper, dask_spikes, win_len, sg_win, dtype=np.float64)
    
    # Use the updated nameit function
    input_file_name = os.path.basename(calcium_signal)
    output_name_feed = ut.nameit(input_file_name, "pre_processed", sg_win=sg_win)
    output_zarr_file = os.path.join(os.getcwd(), "data", output_name_feed)
    print(output_zarr_file)
    
    preprocessed_feed.to_zarr(output_zarr_file, overwrite=True)
    print(f"Processed data saved to {output_zarr_file}")

    output_name_slopes = ut.nameit(input_file_name, "slopes", sg_win=sg_win)
    output_zarr_file_slopes = os.path.join(os.getcwd(), "data", output_name_slopes)
    print(output_zarr_file_slopes)
    slopes.to_zarr(output_zarr_file_slopes, overwrite=True)
    print(f"Slopes data saved to {output_zarr_file_slopes}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <calcium_signal_path> <sg_delta>")
        sys.exit(1)

    # Read arguments
    calcium_signal = sys.argv[1]
    spikes_trains = sys.argv[2]
    sg_win = float(sys.argv[3])

    print(f"Calcium signal address: {calcium_signal}")
    
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
    main(calcium_signal, spikes_trains, sg_win)
