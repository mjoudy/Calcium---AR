#!/bin/bash
# Wrap-up run at N=1250 in the CLEAN asynchronous-irregular regime (regime-scan
# winner: g=8, eta=1.0, J_ex=0.8 -> ~8 Hz, CV 0.80). Same pipeline as the n1250
# baseline (g=5, eta=2); the comparison tests whether the realistic low-rate AI
# regime improves excitatory inference.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_wrapup_n1250ai.slurm

#SBATCH --job-name=ca_wrapup_n1250ai
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/wrapup_n1250ai_%j.log

set -eo pipefail

WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"

export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

GIT_COMMIT=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then GIT_DIRTY=yes; else GIT_DIRTY=no; fi
echo "host: $(hostname)   git: $GIT_COMMIT (dirty: $GIT_DIRTY)   date: $(date)"

# 5 seeds -> results/wrapup_n1250ai/seed{1..5}/ under the workspace
"$PY" scripts/wrapup_run.py --net n1250ai --seeds 1 2 3 4 5

# figures straight from the saved matrices
"$PY" scripts/wrapup_figures.py --data "$WS/results/wrapup_n1250ai" --decision-method endale
echo "done: $(date)"
