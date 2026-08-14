#!/bin/bash
# R.4 scaling ladder, N=12500: seeds 2-5 at the EXISTING 10M-ms ceiling
# (NOT the 20M-ms extension -- that needs 650G+, which no node on this
# cluster has; see run_r4_n12500_extend20m_pilot.slurm's comments and the
# 2026-08-14 notebook entry). N=12500 currently has real data up to 10M ms
# but only 1 seed at every checkpoint, so fig_R4/fig_R5 show no error band
# for it. This fills that in at a T that's already measured to fit
# (10M ms ~= 5-6h/seed, ~400G/seed -- both proven by the existing seed1 run
# and by run_r4_n12500_moredata.slurm), instead of the untested 20M-ms/650G
# target.
#
# One array task per seed (2,3,4,5) so they run concurrently and one
# failing doesn't take the others down.
#
# SIMULATION ONLY -- run analyze_run.py / regenerate fig_R4 + fig_R5 once
# all 4 finish (alongside seed1, which already exists).
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r4_n12500_seeds2to5_10m.slurm

#SBATCH --job-name=ca_r4_n12500_s2to5_10m
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --array=2-5
#SBATCH --output=slurm_logs/r4_n12500_s%a_10m_%j.log

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
assert torch.cuda.is_available(), "No CUDA torch -- set GPU_PY to a CUDA env."
PYCHK

"$PY" scripts/wrapup_run_stream.py --net n12500_r4 \
      --sweep 100000 200000 500000 1000000 2000000 5000000 7500000 10000000 \
      --seeds "$SEED" \
      --methods ols --device cuda --chunk-ms 5000 \
      --cache-dir "$WS/gt_cache" --save-moments

echo "seed $SEED done: $(date)"
