#!/bin/bash
# R.4b FIXED IN-DEGREE ladder, N=2500 — alternative to the main R.4 ladder's
# fixed connection-PROBABILITY (eps=0.1, C_E growing with N). Here C_E is held
# at 100 (n1250_r4's own value) at every size instead, so epsilon shrinks
# (0.1 -> 0.05 at this N). J stays at n1250_r4's 0.474 too (no rescaling
# needed once C_E is fixed). Tests whether the excitatory-recall gap found in
# R.4 is a real, unavoidable effect of network size, or an artifact of the
# fixed-probability scaling convention specifically.
#
# eta=1.30 (n1250_r4's tuned value) is a THEORETICAL prediction for this N
# under fixed in-degree, not yet measured -- fixed C_E + fixed J should give
# near-N-invariant local dynamics (that's the point of this ladder), but the
# preflight below actually checks it before committing to the full run.
# N=1250 needs no new run here -- fixed-eps=0.1 IS fixed-C_E=100 at this one
# size, so n1250_r4's existing data is the shared anchor point for both ladders.
#
# 5 seeds for error bars, matching the main R.4 ladder.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r4ci_n2500.slurm

#SBATCH --job-name=ca_r4ci_n2500
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/r4ci_n2500_%j.log

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
echo "=== regime preflight: N=2500, fixed C_E=100 (eps=0.05), g=8, eta=1.30 ==="
"$PY" scripts/r4_tune_regime.py --sizes 2500 --g 8 --j-scale 1.5 --fixed-ce 100 \
      --etas 1.1 1.2 1.3 1.4 1.5 --target 14 \
      --n-threads "${SLURM_CPUS_PER_TASK}" || true
echo "^^^ if the eta=1.30 row isn't close to 14Hz, STOP and re-tune before"
echo "    trusting the run below (edit n2500_ci's eta in wrapup_run.py)."

"$PY" scripts/wrapup_run_stream.py --net n2500_ci \
      --sweep 100000 200000 500000 1000000 2000000 5000000 --seeds 1 2 3 4 5 \
      --methods ols --device cuda --chunk-ms 5000 \
      --cache-dir "$WS/gt_cache" --save-moments

for Tk in 100 200 500 1000 2000 5000; do
    "$PY" scripts/analyze_run.py --data "$WS/results/wrapup_n2500ci_T${Tk}k" \
        || echo "analyze T${Tk}k failed (continuing)"
done

echo "done: $(date)"
echo "--- copy home just these: ---"
ls -lh "$WS"/results/wrapup_n2500ci_T*k/metrics.csv 2>/dev/null
