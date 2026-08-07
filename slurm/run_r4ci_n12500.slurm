#!/bin/bash
# R.4b FIXED IN-DEGREE ladder, N=12500 — see run_r4ci_n2500.slurm for the full
# explanation. C_E held at 100 (epsilon = 100/10000 = 0.01 at this N -- 10x
# sparser than the main R.4 ladder's 0.1), J fixed at 0.474, eta=1.30 is the
# theoretical prediction (preflight-checked below, not blocking).
#
# This is the size where the two ladders diverge MOST (eps 0.1 vs 0.01), so
# it's also the most important point to get right -- and the most expensive.
#
# 3 SEEDS (not 5, same cost-control reasoning as run_r4_n12500_moredata.slurm):
# N=12500 is the expensive size, and this ladder already needs 3 fresh N=12500
# simulations on top of everything else queued. Bump to 5 if you want full
# error bars here too once you've seen whether the mean trend alone answers
# the question.
#
# COST: same order as the main N=12500 job (~2.5-3h/seed extrapolated), so
# ~3 seeds ~= 8-9h.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r4ci_n12500.slurm

#SBATCH --job-name=ca_r4ci_n12500
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=16:00:00
#SBATCH --output=slurm_logs/r4ci_n12500_%j.log

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
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available(), "No CUDA torch — set GPU_PY to a CUDA env."
PYCHK

# --- preflight: verify eta=1.30 still gives ~14Hz under fixed C_E=100 -------- #
echo "=== regime preflight: N=12500, fixed C_E=100 (eps=0.01), g=8, eta=1.30 ==="
"$PY" scripts/r4_tune_regime.py --sizes 12500 --g 8 --j-scale 1.5 --fixed-ce 100 \
      --etas 1.1 1.2 1.3 1.4 1.5 --target 14 \
      --n-threads "${SLURM_CPUS_PER_TASK}" || true
echo "^^^ if the eta=1.30 row isn't close to 14Hz, STOP and re-tune before"
echo "    trusting the run below (edit n12500_ci's eta in wrapup_run.py)."

"$PY" scripts/wrapup_run_stream.py --net n12500_ci \
      --sweep 100000 200000 500000 1000000 2000000 5000000 --seeds 1 2 3 \
      --methods ols --device cuda --chunk-ms 5000 \
      --cache-dir "$WS/gt_cache" --save-moments

for Tk in 100 200 500 1000 2000 5000; do
    "$PY" scripts/analyze_run.py --data "$WS/results/wrapup_n12500ci_T${Tk}k" \
        || echo "analyze T${Tk}k failed (continuing)"
done

echo "done: $(date)"
echo "--- copy home just these: ---"
ls -lh "$WS"/results/wrapup_n12500ci_T*k/metrics.csv 2>/dev/null
