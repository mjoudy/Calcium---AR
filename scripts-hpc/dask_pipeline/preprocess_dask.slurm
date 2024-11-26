#!/bin/bash

#SBATCH --job-name=preprocess_dask
#SBATCH --output=preprocess_dask.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --partition=dev_single

# Load necessary modules and activate the Conda environment
module load devel/miniconda
module load compiler/gnu/8.5.0
conda activate phd_conda

# Define the project directory (use environment variable or default to current directory)
PROJECT_DIR=${PROJECT_DIR:-$(pwd)}

# Input file (relative to project directory)
INPUT_CALCIUM="$PROJECT_DIR/data/calcium-N1250-T10e06-tau50.0.zarr"
INPUT_SPIKES="$PROJECT_DIR/data/spikes-N1250-T10e06.zarr"

# Array of sg_delta values
SG_WIN=31

# Get the current sg_delta value based on the SLURM array task ID
# INDEX=$SLURM_ARRAY_TASK_ID
# SG_DELTA=${SG_DELTA_VALUES[$INDEX]}

# Run the preprocess_dask.py script with the input file and sg_delta value
python preprocess_dask.py $INPUT_CALCIUM $INPUT_SPIKES $SG_WIN
