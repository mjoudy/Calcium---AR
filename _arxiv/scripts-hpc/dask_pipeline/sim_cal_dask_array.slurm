#!/bin/bash

#SBATCH --job-name=sim_calsium
#SBATCH --output=sim_cal_dask_array.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --partition=dev_single
#SBATCH --array=0-9%1

module load devel/miniconda
module load compiler/gnu/8.5.0
conda activate phd_conda

PROJECT_DIR=${PROJECT_DIR:-$(pwd)}  # Use $PROJECT_DIR if set, else current directory
INPUT_FILE="$PROJECT_DIR/data/spikes-N1250-T10e06.zarr"

TAU_VALUES=(1 10 20 50 100 200 500 1000 2000 3000)
INDEX=$SLURM_ARRAY_TASK_ID
TAU=${TAU_VALUES[$INDEX]}

#srun python sim_cal.py $TAU
python sim_cal_dask_array.py $TAU

