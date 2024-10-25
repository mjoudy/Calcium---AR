# run_simulation.py

import os
import subprocess
import h5py
import numpy as np
import brunel_network as bn
import json
import nest
import sys

print('Running Brunel Network Simulation...')

# Initialize and run the simulation
ground_truth = bn.BrunelNetwork()
ground_truth.setup_kernel()
ground_truth.setup_elements()
ground_truth.connect_elements()
ground_truth.simulate()

# Record and save data
ground_truth.record_data()
sp_file, conn_file = ground_truth.save_data()

#env = os.environ.copy()
#os.environ["SP_FILE"] = sp_file
#os.environ["CONN_FILE"] = conn_file
print('Simulation completed. Data saved to files:')
print(f'Spike trains file: {sp_file}')
print(f'Connections file: {conn_file}')
#test = os.getenv("SP_FILE")
#print(test)

#subprocess.run(["python", "pipeline_dask.py"], env=env)

file_names = {
    "sp_file": sp_file,
    "conn_file": conn_file
}

with open("temp_names.json", "w") as f:
    json.dump(file_names, f)

nest.ResetKernel()
sys.exit()
