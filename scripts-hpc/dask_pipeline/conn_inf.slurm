#!/bin/bash

#SBATCH --job-name=conn_inf
#SBATCH --output=conn_inf.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=170G
#SBATCH --time=00:30:00
#SBATCH --partition=dev_single

# Load necessary modules and activate the Conda environment
module load devel/miniconda
module load compiler/gnu/8.5.0
conda activate phd_conda

# Define the project directory (use environment variable or default to current directory)
PROJECT_DIR=${PROJECT_DIR:-$(pwd)}

INPUT_SIGNALS="$PROJECT_DIR/data/pre_processed-N1250-T10e06-tau50.0-sg_win31.zarr"
INPUT_CONN_MAT="$PROJECT_DIR/data/connectivity-N1250-T10e06.npy"

INPUT_LAG=10

python conn_inf.py $INPUT_SIGNALS $INPUT_CONN_MAT $INPUT_LAG

