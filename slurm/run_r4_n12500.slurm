#!/bin/bash
# R.4 scaling ladder, TOP RUNG: N=12500 in the matched AI regime
# (g=8, J=0.150, eta=2.60, V_reset=10  ->  13.9 Hz, CV 1.06, synchrony 0.007).
#
# Its own job because it is the expensive one: ONE 5M-ms simulation per seed,
# scored at SIX checkpoints (100k/200k/500k/1M/2M/5M ms) so this line spans the
# full fig_R4 x-axis, matching N=1250/2500/5000's grid. Each checkpoint is
# solved and SAVED as it is reached, so a timeout still leaves the shorter
# recordings on disk.
#
# 5 SEEDS (was 1): error bars for fig_R4. Seed 1's checkpoints are already
# cached (this job originally just added the 3 short ones for free -- see the
# "cache reuse" note in scripts/wrapup_run_stream.py), but seeds 2-5 are NEW
# and each needs a full fresh 5M-ms NEST simulation -- there's no shortcut for
# those. This is the expensive one: extrapolating from observed costs at
# smaller N (N=1250:16min, N=2500:32min, N=5000:66min, roughly linear in N),
# N=12500 at 5M ms is ~2.5-3h per seed, so 4 new seeds ~= 10-12h. Time budget
# below is a safety margin on top of that extrapolation (not yet measured
# directly for N=12500) -- if your partition caps job time below this, split
# seeds 2-5 into separate jobs instead of one.
#
# NOTE this replaces the reuse of wrapup_n12500lr (g=6, J=0.1): that run is a
# different regime family (CV 0.62) and is no longer part of this ladder. It
# stays on disk as the canonical-vs-tuned comparison for R.6.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r4_n12500.slurm

#SBATCH --job-name=ca_r4_n12500
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/r4_n12500_%j.log
#
# 240G: spike events at ~14 Hz x 12500 neurons x 5000 s is ~875M events per
# seed; processed sequentially so peak memory doesn't grow with seed count.
# 24 h: ~10-12h extrapolated for the 4 new seeds, generous margin on top.

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
print("gpu:", torch.cuda.get_device_name(0))
PYCHK

# --- preflight: confirm the regime BEFORE the 12 h part starts ---------------
# Re-measures exactly the config the sweep is about to run (4 s sim, seconds of
# work). Expect ~13.9 Hz, CV ~1.06, sync ~0.007.
echo "=== regime preflight: N=12500, g=8, J x1.5, eta=2.60 ==="
"$PY" scripts/r4_tune_regime.py --sizes 12500 --g 8 --j-scale 1.5 \
      --etas 2.6 --target 14 --n-threads "${SLURM_CPUS_PER_TASK}" || true

"$PY" scripts/wrapup_run_stream.py --net n12500_r4 \
      --sweep 100000 200000 500000 1000000 2000000 5000000 --seeds 1 2 3 4 5 \
      --methods ols --device cuda --chunk-ms 5000 \
      --cache-dir "$WS/gt_cache" --save-moments

# score on the cluster; only the small metrics.csv files travel home
for Tk in 100 200 500 1000 2000 5000; do
    "$PY" scripts/analyze_run.py --data "$WS/results/wrapup_n12500r4_T${Tk}k" \
        || echo "analyze T${Tk}k failed (continuing)"
done

echo "done: $(date)"
echo "--- copy home just these: ---"
ls -lh "$WS"/results/wrapup_n12500r4_T*k/metrics.csv 2>/dev/null
