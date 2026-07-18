#!/bin/bash
# Stage 2b — pure-Lasso vs Elastic-Net L1/L2 sweep at the N=1250 operating point.
# Answers: does pure Lasso help precision? does the L1/L2 ratio matter? and the
# FAIR (density-matched) cross-method comparison.
#
# Prereq: the reference dataset $WS/data/N1250/s1_tau100_T50000 exists (from prep).
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_lasso_sweep.slurm

#SBATCH --job-name=ca_lasso_sweep
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/lasso_sweep_%j.log

set -eo pipefail

WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"

export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

GIT_COMMIT=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then GIT_DIRTY=yes; else GIT_DIRTY=no; fi
echo "host: $(hostname)   git: $GIT_COMMIT (dirty: $GIT_DIRTY)   date: $(date)"

"$PY" scripts/run_lasso_sweep.py
