#!/bin/bash
# R.4b FIXED IN-DEGREE ladder, N=5000 — see run_r4ci_n2500.slurm for the full
# explanation. C_E held at 100 (epsilon = 100/4000 = 0.025 at this N), J fixed
# at 0.474, eta=1.30 is the theoretical prediction (preflight-checked below,
# not blocking).
#
# 5 seeds for error bars, matching the main R.4 ladder.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r4ci_n5000.slurm

#SBATCH --job-name=ca_r4ci_n5000
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/r4ci_n5000_%j.log

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
PYCHK

# --- preflight: verify eta=1.30 still gives ~14Hz under fixed C_E=100 -------- #
echo "=== regime preflight: N=5000, fixed C_E=100 (eps=0.025), g=8, eta=1.30 ==="
"$PY" scripts/r4_tune_regime.py --sizes 5000 --g 8 --j-scale 1.5 --fixed-ce 100 \
      --etas 1.1 1.2 1.3 1.4 1.5 --target 14 \
      --n-threads "${SLURM_CPUS_PER_TASK}" || true
echo "^^^ if the eta=1.30 row isn't close to 14Hz, STOP and re-tune before"
echo "    trusting the run below (edit n5000_ci's eta in wrapup_run.py)."

"$PY" scripts/wrapup_run_stream.py --net n5000_ci \
      --sweep 100000 200000 500000 1000000 2000000 5000000 --seeds 1 2 3 4 5 \
      --methods ols --device cuda --chunk-ms 5000 \
      --cache-dir "$WS/gt_cache" --save-moments

for Tk in 100 200 500 1000 2000 5000; do
    "$PY" scripts/analyze_run.py --data "$WS/results/wrapup_n5000ci_T${Tk}k" \
        || echo "analyze T${Tk}k failed (continuing)"
done

echo "done: $(date)"
echo "--- copy home just these: ---"
ls -lh "$WS"/results/wrapup_n5000ci_T*k/metrics.csv 2>/dev/null
