#!/bin/bash
# Run the N=1250 experiment on NEMO2 (CPU partition).
#
# Submit from inside the repo on the cluster:
#     cd $(ws_find calcium_ar)/Calcium---AR
#     sbatch slurm/run_n1250.slurm
#
# Check progress:
#     squeue --me
#     tail -f slurm_logs/n1250_<jobid>.log

#SBATCH --job-name=ca_n1250
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           # MUST match n_threads in scripts/run_n1250.py
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/n1250_%j.log

set -eo pipefail

# --------------------------------------------------------------------------- #
# Environment: load conda, activate the env that lives in the workspace.
# --------------------------------------------------------------------------- #
module load conda
WS=$(ws_find calcium_ar)

cd "$WS/Calcium---AR"

export CALCIUM_AR_WORKDIR="$WS"                 # big outputs go to the workspace
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" # match BLAS threads to allocated cores

# --------------------------------------------------------------------------- #
# Reproducibility header (lands at the top of the log).
# --------------------------------------------------------------------------- #
GIT_COMMIT=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then GIT_DIRTY=yes; else GIT_DIRTY=no; fi
echo "================== run info =================="
echo "host:    $(hostname)"
echo "date:    $(date)"
echo "git:     $GIT_COMMIT (dirty: $GIT_DIRTY)"
echo "cpus:    ${SLURM_CPUS_PER_TASK}"
echo "workdir: $WS"
echo "============================================="

# --------------------------------------------------------------------------- #
# Run. 'conda run' reliably activates the workspace env in a non-interactive
# job — plain 'conda activate' often does not take effect in batch shells.
# --------------------------------------------------------------------------- #
conda run --no-capture-output -p "$WS/env" python scripts/run_n1250.py
