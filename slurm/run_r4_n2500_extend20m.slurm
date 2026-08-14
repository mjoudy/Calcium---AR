#!/bin/bash
# R.4 scaling ladder, N=2500: EXTEND to the full 10-point grid, matching
# N=1250's ceiling (100k/200k/500k/1M/2M/5M/7.5M/10M/15M/20M ms) instead of
# stopping at 5M ms. 2026-08-14 decision: unify every size onto one grid so
# fig_R4's curves are directly comparable end to end, and drop the old
# single-seed 400k probe (now excluded by fig_r4_plot.py's GRID_MS filter).
#
# 5 SEEDS, full fresh resimulation (extending max_T is NOT incremental --
# checkpointing within one run is free, but a longer run can't reuse a
# shorter one's cache). Observed: 32 min/seed at 5M ms (2026-08-06) -> ~128
# min/seed at 20M ms (roughly linear in T) -> 5 seeds ~= 10.7h of compute.
# Budget below has a safety margin on top of that extrapolation.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r4_n2500_extend20m.slurm

#SBATCH --job-name=ca_r4_n2500x20m
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --output=slurm_logs/r4_n2500x20m_%j.log
#
# 256G: ~4x the 64G used for the old 5M-ms/5-seed run (20M ms is 4x longer ->
# ~4x the spike-event count). Bump if it OOMs.

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

"$PY" scripts/wrapup_run_stream.py --net n2500_r4 \
      --sweep 100000 200000 500000 1000000 2000000 5000000 7500000 10000000 \
               15000000 20000000 \
      --seeds 1 2 3 4 5 \
      --methods ols --device cuda --chunk-ms 5000 \
      --cache-dir "$WS/gt_cache" --save-moments

for Tk in 100 200 500 1000 2000 5000 7500 10000 15000 20000; do
    "$PY" scripts/analyze_run.py --data "$WS/results/wrapup_n2500r4_T${Tk}k" \
        || echo "analyze T${Tk}k failed (continuing)"
done

echo "done: $(date)"
echo "--- copy home just these: ---"
ls -lh "$WS"/results/wrapup_n2500r4_T*k/metrics.csv 2>/dev/null
