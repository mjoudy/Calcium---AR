#!/bin/bash
# R.4 follow-up: does N=12500's excitatory-recall gap actually PLATEAU, or is
# it just not-yet-converged? We've only ever recorded up to 5M ms for this
# size (T/N=4000) -- push to 10M ms (T/N=8000) and see if the curve visibly
# bends flat or keeps climbing. Directly tests the "structural ceiling" claim
# in the notebook rather than assuming it.
#
# New checkpoints 7.5M/10M ms are ADDED to the existing 6-point grid (checkpointing
# within one run is free), but the max_T changes (5M -> 10M), so this is a FULL
# FRESH resimulation for every seed -- no cache reuse this time.
#
# 3 SEEDS (not 5, unlike the other R.4 jobs): this specific question is about
# the MEAN trend (does the curve bend?), not primarily about variance, and
# N=12500 is the most expensive size in the ladder -- 3 seeds keeps this
# tractable. Bump --seeds below to "1 2 3 4 5" if you want full error bars here
# too, but expect roughly 5/3 x the time.
#
# COST (extrapolated, not measured -- nothing has been run at N=12500 x 10M ms
# before): 5M ms took ~2.5-3h/seed at this size (extrapolated from N=1250/2500/
# 5000's observed 16/32/66 min). Roughly linear in T, so 10M ms ~= 5-6h/seed x
# 3 seeds ~= 15-18h. Generous margin below. If your partition caps job time
# under that, split seeds into separate sbatch calls instead of one job.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r4_n12500_moredata.slurm

#SBATCH --job-name=ca_r4_n12500_more
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=30:00:00
#SBATCH --output=slurm_logs/r4_n12500_more_%j.log
#
# 400G: double the recording length ~= double the spike-event count vs the
# 5M-ms run (which needed ~240G) -- rough scaling, bump if it OOMs.

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

"$PY" scripts/wrapup_run_stream.py --net n12500_r4 \
      --sweep 100000 200000 500000 1000000 2000000 5000000 7500000 10000000 \
      --seeds 1 2 3 \
      --methods ols --device cuda --chunk-ms 5000 \
      --cache-dir "$WS/gt_cache" --save-moments

for Tk in 100 200 500 1000 2000 5000 7500 10000; do
    "$PY" scripts/analyze_run.py --data "$WS/results/wrapup_n12500r4_T${Tk}k" \
        || echo "analyze T${Tk}k failed (continuing)"
done

echo "done: $(date)"
echo "--- copy home just these: ---"
ls -lh "$WS"/results/wrapup_n12500r4_T*k/metrics.csv 2>/dev/null
