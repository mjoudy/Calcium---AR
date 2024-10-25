#!/bin/bash

# Run the simulation script
echo "Running NEST simulation..."
python sim_ground_truth.py

# Check if the simulation completed successfully
if [ $? -ne 0 ]; then
    echo "Simulation failed. Exiting."
    exit 1
fi

echo "NEST simulation completed. Starting Dask processing..."

# Run the Dask processing script
python pipeline_dask.py

# Check if the Dask processing completed successfully
if [ $? -ne 0 ]; then
    echo "Dask processing failed. Exiting."
    exit 1
fi

echo "Dask processing completed successfully."
