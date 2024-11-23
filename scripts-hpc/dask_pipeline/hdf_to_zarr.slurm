#!/bin/bash
#SBATCH --job-name=hdf_to_zarr
#SBATCH --output=hdf_to_zarr.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --partition=dev_single

module load devel/miniconda
module load compiler/gnu/8.5.0
conda activate phd_conda

python hdf_to_zarr.py

