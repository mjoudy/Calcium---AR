#!/bin/bash
#SBATCH --job-name=sim_gt
#SBATCH --output=sim_gt_dev_single.txt
#SBATCH --error=sim_gt_error.txt
#SBATCH --partition=dev_single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00
#SBATCH --mem=160G

#PROJECT_DIR="$(dirname "$(pwd)")"
#python_script="$PROJECT_DIR/sim_ground_truth.py"

#echo "Project directory (one level up) is: $PROJECT_DIR"
#echo "Python script path is: $python_script"

module load devel/miniconda
module load compiler/gnu/8.5.0
conda activate phd_conda

#python sim_ground_truth.py
#srun --mpi=pmix python sim_ground_truth.py
python sim_ground_truth.py

