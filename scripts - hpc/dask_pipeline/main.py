import subprocess
import time

# Run the NEST simulation part
subprocess.run(["python", "sim_ground_truth.py"], check=True)
time.sleep(10)  # Brief pause to release resources

# Run the Dask processing part
subprocess.run(["python", "pipeline_dask.py"], check=True)