import dask.array as da
from dask.distributed import Client
import numpy as np
import matplotlib.pyplot as plt
from functions import kernel_est_funcs as kef
from functions import conn_inf_funcs as cif
import h5py
import json
import os

def main():
    # Loading file names from temp_names.json
    with open("temp_names.json", "r") as f:
        file_names = json.load(f)

    sp_file = file_names["sp_file"]
    conn_file = file_names["conn_file"]
    print(f"SP_FILE: {sp_file}")
    print(f"CONN_FILE: {conn_file}")

    # Loading spike train data
    spikes_hdf = h5py.File(sp_file, 'r')
    spikes_trains = spikes_hdf['spikes_trains']

    # Initializing Dask client
    client = Client()
    dashboard_link = client.dashboard_link
    print(f"Dask Dashboard: {dashboard_link}")

    # Loading and processing data
    rec_length = json.load(open('network_config.json'))['sim_params']['sim_length']
    dask_array = da.from_array(spikes_trains, chunks=('auto', rec_length))
    pr1 = dask_array.map_blocks(kef.dask_calcium, dtype=dask_array.dtype)
    pr2 = pr1.map_blocks(kef.dask_smooth, dtype=dask_array.dtype)
    result = pr2.compute()

    # Calculating connectivity and saving result
    corr_coef, est_conn = cif.conn_inf_LR(conn_file, result)
    print(corr_coef)
    np.save('est_' + conn_file, est_conn)

if __name__ == "__main__":
    main()
