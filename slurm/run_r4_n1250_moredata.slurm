#!/bin/bash
# R.4 follow-up: push the SMALLEST (least-confounded) network to much more
# data, to find where its excitatory-recall/precision genuinely maxes out --
# a positive-control confusion matrix showing the method CAN work very well
# when the shared-input confound is weak and data isn't the bottleneck.
# N=1250 is by far the cheapest size, so this can push much further than the
# N=12500 companion job (run_r4_n12500_moredata.slurm): up to 20M ms (T/N=16000,
# 4x the current max).
#
# New checkpoints 7.5M/10M/15M/20M ms ADDED to the existing 6-point grid
# (checkpointing is free), but max_T changes (5M -> 20M) so this is a full
# fresh resimulation for every seed.
#
# 5 SEEDS: cheap enough here to keep full error bars.
#
# COST (extrapolated from observed N=1250 @ 5M ms = 16 min/seed, roughly
# linear in T): 20M ms ~= 4x = ~64 min/seed x 5 seeds ~= 5.3h total. Generous
# margin in the time budget below.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r4_n1250_moredata.slurm

#SBATCH --job-name=ca_r4_n1250_more
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/r4_n1250_more_%j.log

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

"$PY" scripts/wrapup_run_stream.py --net n1250_r4 \
      --sweep 100000 200000 500000 1000000 2000000 5000000 7500000 10000000 15000000 20000000 \
      --seeds 1 2 3 4 5 \
      --methods ols --device cuda --chunk-ms 5000 \
      --cache-dir "$WS/gt_cache" --save-moments

for Tk in 100 200 500 1000 2000 5000 7500 10000 15000 20000; do
    "$PY" scripts/analyze_run.py --data "$WS/results/wrapup_n1250r4_T${Tk}k" \
        || echo "analyze T${Tk}k failed (continuing)"
done

echo "done: $(date)"
echo "--- copy home just these: ---"
ls -lh "$WS"/results/wrapup_n1250r4_T*k/metrics.csv 2>/dev/null
