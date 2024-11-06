import dask.array as da
from dask.distributed import Client
import webbrowser
import numpy as np

from functions import dask_functions as df
from functions import utils as ut

import h5py
import json

def main():

    tau = 100
    wup_time = 100

    with open('temp_names.json', 'r') as file:
        file_names = json.load(file)
        sp_name = file_names['sp_file']

    with h5py.File('data/'+sp_name, 'r') as f:
        spikes_trains = f['spikes_trains'][wup_time:]

    client = Client()
    dashboard_link = client.dashboard_link
    print(f"Dask Dashboard: {dashboard_link}")
    webbrowser.open(dashboard_link)

    with open('network_config.json', 'r') as config_file:
        rec_length = json.load(config_file)['sim_params']['sim_length']
    
    dask_spikes = da.from_array(spikes_trains, chunks=('auto', rec_length))

    pr1 = dask_spikes.map_blocks(df.dask_calcium, tau, dtype=dask_spikes.dtype)
    pr_feed = pr1.map_blocks(df.dask_feed_raper, spikes_trains, dtype=spikes_trains.dtype)
    pr_b_fits = pr1.map_blocks(df.dask_fits_raper, spikes_trains, dtype=float)

    result = pr_feed.compute()
    b_fits = pr_b_fits.compute()

    processed_name = ut.nameit('processed', tau=tau)
    
    with h5py.File('data/'+processed_name, 'w') as h5f:
        h5f.create_dataset('feed', data=result)
        h5f.create_dataset('b_fits', data=b_fits)

    print(f"Processed data saved as {processed_name}")

if __name__ == "__main__":
    main()

    