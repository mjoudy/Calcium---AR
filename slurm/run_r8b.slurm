#!/bin/bash
# R.8 phase 2 (multiple lags) — the TIMING axis at N=12500. Reads the cached
# spikes (no NEST), streams the multi-lag moments C(l) for a grid of lags,
# forms per-lag connectivity A_l = C(l) C(0)^-1, and saves the per-pair timing
# signature + symmetry so the two axes can be analysed separately.
#
# The multi-lag accumulation is ~len(lags)x the single-lag matmul cost, so this
# is a semi-long GPU job (a few hours at 5M ms). Only a small npz travels home.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r8b.slurm

#SBATCH --job-name=ca_r8b
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/r8b_%j.log

set -eo pipefail

WS=$(ws_find calcium_ar)
PY="${GPU_PY:-$WS/env-gpu/bin/python}"
[ -x "$PY" ] || PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"
export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "host: $(hostname)   git: $(git rev-parse HEAD)   date: $(date)   PY=$PY"
"$PY" - <<'PYCHK'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
assert torch.cuda.is_available(), "no CUDA torch — set GPU_PY to a CUDA env."
PYCHK

R8BOUT="$WS/results/r8b"
"$PY" scripts/r8b_multilag.py --net n12500_r4 --cache-dir "$WS/gt_cache" \
      --checkpoint-ms 5000000 --chunk-ms 5000 --device cuda --out-dir "$R8BOUT"

echo "done: $(date)"
ls -lh "$R8BOUT"/*.npz 2>/dev/null || echo "(no npz)"
echo
echo "rsync -avhz ${USER}@login1.nemo.uni-freiburg.de:$R8BOUT/ ~/calcium_results/r8b/"
