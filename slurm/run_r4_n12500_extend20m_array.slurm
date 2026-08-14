#!/bin/bash
# R.4 scaling ladder, N=12500: seeds 2-5 for the 20M-ms extension.
#
# DO NOT SUBMIT until run_r4_n12500_extend20m_pilot.slurm (seed 1) has
# actually finished successfully -- this confirms the time/memory budgets
# below (extrapolated, unmeasured at this length) are workable before
# committing 4x the compute to them.
#
# One SLURM job ARRAY task per seed (2,3,4,5) so they can run concurrently if
# the partition has capacity, and so one seed failing doesn't take the others
# down with it. Same per-seed cost estimate as the pilot (~10-12h, ~650G).
#
# SIMULATION ONLY, same reasoning as the pilot -- do not call analyze_run.py
# here (would race across concurrently-finishing array tasks). Run
# slurm/run_r4_n12500_extend20m_analyze.slurm once every seed (1-5) is done.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r4_n12500_extend20m_array.slurm

#SBATCH --job-name=ca_r4_n12500x20m
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=650G
#SBATCH --time=20:00:00
#SBATCH --array=2-5
#SBATCH --output=slurm_logs/r4_n12500x20m_s%a_%j.log

set -eo pipefail

WS=$(ws_find calcium_ar)
PY="${GPU_PY:-$WS/env-gpu/bin/python}"
[ -x "$PY" ] || PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"

export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

SEED="${SLURM_ARRAY_TASK_ID}"
echo "host: $(hostname)  git: $(git rev-parse HEAD)  date: $(date)  PY=$PY  seed=$SEED"
"$PY" - <<'PYCHK'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available(), "No CUDA torch — set GPU_PY to a CUDA env."
PYCHK

"$PY" scripts/wrapup_run_stream.py --net n12500_r4 \
      --sweep 100000 200000 500000 1000000 2000000 5000000 7500000 10000000 \
               15000000 20000000 \
      --seeds "$SEED" \
      --methods ols --device cuda --chunk-ms 5000 \
      --cache-dir "$WS/gt_cache" --save-moments

echo "seed $SEED done: $(date)"
