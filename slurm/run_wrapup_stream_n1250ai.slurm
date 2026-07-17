#!/bin/bash
# Data-length sweep in the clean-AI regime (n1250ai): does MORE recording recover
# excitatory inference where correlations are weak? Streaming engine keeps RAM at
# O(N^2): sparse spikes, chunked calcium/feed, moment accumulation, snapshots at
# each recording length. One long sim per seed yields the whole curve.
#
# Sweep: 500k / 1M / 2M ms  (10x / 20x / 40x the 50k baseline).
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_wrapup_stream_n1250ai.slurm

#SBATCH --job-name=ca_stream_n1250ai
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/stream_n1250ai_%j.log

set -eo pipefail

WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"

export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

GIT_COMMIT=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then GIT_DIRTY=yes; else GIT_DIRTY=no; fi
echo "host: $(hostname)   git: $GIT_COMMIT (dirty: $GIT_DIRTY)   date: $(date)"

# 3 seeds x {500k,1M,2M} ms -> results/wrapup_n1250ai_T{500,1000,2000}k/
"$PY" scripts/wrapup_run_stream.py --net n1250ai \
      --sweep 500000 1000000 2000000 --seeds 1 2 3 --chunk-ms 10000

# figures per recording length
for Tk in 500 1000 2000; do
    "$PY" scripts/wrapup_figures.py --data "$WS/results/wrapup_n1250ai_T${Tk}k" \
          --decision-method endale || echo "figure T${Tk}k failed (continuing)"
done
echo "done: $(date)"
