#!/bin/bash
# Wrap-up run at N=1250 (validated network): all five methods on identical data,
# 5 seeds, scored with the density-threshold metric set. Produces the per-seed
# estimates the wrap-up figures read.
#
# Pipeline per seed: build 1250-neuron Brunel net -> shared deconvolved feed ->
# OLS, EN, Lasso, Lasso+Dale, EN+Dale. Balance is intentionally skipped.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_wrapup_n1250.slurm
#
# Then build the figures (cheap, from the saved matrices):
#          "$PY" scripts/wrapup_figures.py --data "$WS/results/wrapup_n1250"

#SBATCH --job-name=ca_wrapup_n1250
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/wrapup_n1250_%j.log

set -eo pipefail

WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"

export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

GIT_COMMIT=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then GIT_DIRTY=yes; else GIT_DIRTY=no; fi
echo "host: $(hostname)   git: $GIT_COMMIT (dirty: $GIT_DIRTY)   date: $(date)"

# 5 seeds -> results/wrapup_n1250/seed{1..5}/ under the workspace
"$PY" scripts/wrapup_run.py --net n1250 --seeds 1 2 3 4 5

# Figures render straight from the saved matrices (light; keep the CPU job simple)
"$PY" scripts/wrapup_figures.py --data "$WS/results/wrapup_n1250" --decision-method endale
echo "done: $(date)"
