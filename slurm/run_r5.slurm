#!/bin/bash
# R.5 (regularization vs data) — one GPU job, both sizes.
#
# Two phases, no new NEST simulation (both networks are already cached as spike
# events in gt_cache, so streaming reads them instead of re-simulating):
#
#   1. EXTEND the checkpoint set into the data-scarce regime. R.4 only reached
#      T/N = 800; the "cure" from regularization is expected at smaller T/N, so
#      we add T/N = 100 / 200 / 400 and re-save the moments there.
#         N= 1250 -> 12.5k / 25k / 50k ms
#         N=12500 -> 125k  / 250k / 500k ms
#
#   2. SOLVE the lambda sweep from the cached moments (existing 800/1600/4000 +
#      the new 100/200/400) for all six estimators, and score.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r5.slurm

#SBATCH --job-name=ca_r5
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/r5_%j.log

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

# --- phase 1: short-T moments (reads cached spikes, --save-moments) ----------
echo "=== phase 1: extend checkpoints into the data-scarce regime ==="
"$PY" scripts/wrapup_run_stream.py --net n1250_r4  --sweep 12500 25000 50000 \
      --seeds 1 --methods ols --chunk-ms 5000 \
      --cache-dir "$WS/gt_cache" --save-moments
"$PY" scripts/wrapup_run_stream.py --net n12500_r4 --sweep 125000 250000 500000 \
      --seeds 1 --methods ols --device cuda --chunk-ms 5000 \
      --cache-dir "$WS/gt_cache" --save-moments

# --- phase 2: lambda sweep + scoring from the cached moments ------------------
echo "=== phase 2: R.5 solve + score ==="
"$PY" scripts/fig_r5_compute.py --root "$WS/results" --net n1250r4  --device cpu
"$PY" scripts/fig_r5_compute.py --root "$WS/results" --net n12500r4 --device cuda

echo "done: $(date)"
echo "--- copy home just these: ---"
ls -lh "$WS"/results/r5/*.csv 2>/dev/null
echo
echo "rsync -avhz ${USER}@login1.nemo.uni-freiburg.de:$WS/results/r5/ ~/calcium_results/r5/"
