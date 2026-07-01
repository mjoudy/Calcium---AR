#!/bin/bash
# Parameter sweeps at N=1250 (EN -> Dale -> balance). One file; pick the sweep at
# submit time via SWEEP, and set --array to match its size.
#
# Run PREP first (generates every dataset, one per task), WAIT for it to finish,
# then run the sweeps (they only LOAD the datasets):
#
#   cd $(ws_find calcium_ar)/Calcium---AR
#   sbatch --export=ALL,SWEEP=prep --array=0-54 slurm/run_experiments.slurm
#   # wait until prep is done (squeue --me shows nothing), then:
#   sbatch --export=ALL,SWEEP=lag  --array=0-59 slurm/run_experiments.slurm
#   sbatch --export=ALL,SWEEP=lam  --array=0-29 slurm/run_experiments.slurm
#   sbatch --export=ALL,SWEEP=tau  --array=0-34 slurm/run_experiments.slurm
#   sbatch --export=ALL,SWEEP=data --array=0-24 slurm/run_experiments.slurm
#
# (Safety: the sweeps refuse to run if their dataset is missing, so an early
#  submit fails cleanly instead of racing prep.)

#SBATCH --job-name=ca_exp
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           # MUST match n_threads in the runner
#SBATCH --mem=160G                  # sized for the largest case (data sweep, T=200k)
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/exp_%A_%a.log

set -eo pipefail

: "${SWEEP:?set SWEEP (prep|lag|lam|tau|data) via --export=ALL,SWEEP=...}"

WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"     # call the env's python directly (most reliable in batch)
cd "$WS/Calcium---AR"

export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

GIT_COMMIT=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then GIT_DIRTY=yes; else GIT_DIRTY=no; fi
echo "================== run info =================="
echo "host:    $(hostname)"
echo "date:    $(date)"
echo "git:     $GIT_COMMIT (dirty: $GIT_DIRTY)"
echo "sweep:   ${SWEEP}   task: ${SLURM_ARRAY_TASK_ID}"
echo "cpus:    ${SLURM_CPUS_PER_TASK}"
echo "workdir: $WS"
echo "============================================="

"$PY" scripts/run_n1250_experiments.py
