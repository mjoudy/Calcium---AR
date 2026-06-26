#!/bin/bash
# Groups 1 & 2 solver/strength comparison at N=1250 — 10 runs as a job array,
# all running in parallel across the cpu partition.
#
# Prerequisite: run_n1250.py has already built the shared dataset
#   ($WS/data/N1250). Each array task loads it (no re-simulation).
#
# Submit (from inside the repo on the cluster):
#     cd $(ws_find calcium_ar)/Calcium---AR
#     sbatch slurm/run_n1250_sweep.slurm
# Watch:
#     squeue --me
#     tail -f slurm_logs/sweep_<arrayjobid>_<task>.log

#SBATCH --job-name=ca_sweep
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           # MUST match n_threads in scripts/run_n1250_sweep.py
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-9                 # 10 settings (indices 0..9)
#SBATCH --output=slurm_logs/sweep_%A_%a.log    # %A = array job id, %a = task index

set -eo pipefail

WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"     # call the env's python directly (most reliable in batch)

cd "$WS/Calcium---AR"

export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# --------------------------------------------------------------------------- #
# Reproducibility header (lands at the top of each task's log).
# --------------------------------------------------------------------------- #
GIT_COMMIT=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then GIT_DIRTY=yes; else GIT_DIRTY=no; fi
echo "================== run info =================="
echo "host:    $(hostname)"
echo "date:    $(date)"
echo "git:     $GIT_COMMIT (dirty: $GIT_DIRTY)"
echo "task:    ${SLURM_ARRAY_TASK_ID}"
echo "cpus:    ${SLURM_CPUS_PER_TASK}"
echo "workdir: $WS"
echo "============================================="

"$PY" scripts/run_n1250_sweep.py
