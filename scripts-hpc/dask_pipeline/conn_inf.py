import zarr
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import sys
from joblib import cpu_count
from functions import conn_inf_funcs as cif
from functions import utils as ut

def main(signals, conn_matrix, lag):
    signals = zarr.open(signals)
    conn_matrix = np.load(conn_matrix)

    n_cpu = cpu_count()
    print(f"Number of CPUs: {n_cpu}")

    # Monitor CPU usage before and after execution
    print("Monitoring CPU usage...")
    initial_cpu_usage, available_cpus = ut.monitor_cpu_usage()
    print(f"Initial CPU usage: {initial_cpu_usage}% of {available_cpus} CPUs available.")

    corr, A = cif.conn_inf_LR(conn_matrix, signals, lag)

    final_cpu_usage, available_cpus = ut.monitor_cpu_usage()
    print(f"Final CPU usage: {final_cpu_usage}% of {available_cpus} CPUs available.")

    np.save("output_A.npy", A)
    print(f"Correlation: {corr}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python conn_inf.py <signals.zarr> <conn_matrix.npy>")
        sys.exit(1)

    signals = sys.argv[1]
    conn_matrix = sys.argv[2]
    lag = int(sys.argv[3])
    main(signals, conn_matrix, lag)
