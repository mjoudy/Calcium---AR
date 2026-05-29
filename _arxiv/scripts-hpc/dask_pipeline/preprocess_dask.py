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

def main(calcium_signal, spikes_trains, sg_win, win_len):

    z = zarr.open(calcium_signal, mode="r")
    data_shape = z.shape
    dask_calcium = da.from_zarr(calcium_signal, chunks=("auto", data_shape[1]))
    dask_spikes = da.from_zarr(spikes_trains, chunks=("auto", data_shape[1]))

    preprocessed_feed = da.map_blocks(df.dask_preprocess, 
        dask_calcium, dask_spikes, win_len, sg_win, 
        dtype=np.float64
    )
    
    input_file_name = os.path.basename(calcium_signal)
    output_name_feed = ut.nameit(input_file_name, "pre_processed", sg_win=sg_win)
    output_zarr_file_feed = os.path.join(os.getcwd(), "data", output_name_feed)
    preprocessed_feed.to_zarr(output_zarr_file_feed, overwrite=True)
    print(f"Processed feed data saved to {output_zarr_file_feed}")



if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <calcium_signal_path> <sg_delta>")
        sys.exit(1)

    calcium_signal = sys.argv[1]
    spikes_trains = sys.argv[2]
    sg_win =int(sys.argv[3])
    win_len =int(sys.argv[4])

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

    client = Client(cluster)
    print(cluster)
    print(client)
    
    main(calcium_signal, spikes_trains, sg_win, win_len)
