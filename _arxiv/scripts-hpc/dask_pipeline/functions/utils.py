import json
import os
import psutil

#This function has been created to avoid confiosion of files durind mass experiments.
#One can use it to name files in a way that is easy to understand and identify parameters space and stages of the job(spikes, calcium, preprocessed, ...).
#this function can be easily updated by adding more arguments as parameters of different stages. 
# Generalized nameit function with '-' as the separator
def nameit(input_file, replacement, **kwargs):
    """
    Generate a modified file name by replacing the first word of the base name.

    Parameters:
        input_file (str): The input file name.
        replacement (str): The replacement for the first word in the base name.
        kwargs: Additional parameters to append to the file name.

    Returns:
        str: The modified file name.
    """
    base_name, ext = input_file.rsplit('.zarr', 1)
    parts = base_name.split('-', 1)  # Split at the first '-'
    if len(parts) > 1:
        # Replace the first word
        modified_name = f"{replacement}-{parts[1]}"
    else:
        # If no '-', replace the entire name
        modified_name = replacement

    for key, value in kwargs.items():
        modified_name += f"-{key}{value}"

    modified_name += '.zarr'
    return modified_name

def monitor_cpu_usage():
    """
    Monitor the actual number of CPUs used during runtime.
    """
    process = psutil.Process(os.getpid())
    cpu_usage = process.cpu_percent(interval=1)  # Measure over a 1-second interval
    cpu_count = len(process.cpu_affinity())     # Get CPU affinity (available CPUs)
    return cpu_usage, cpu_count
