#!/bin/bash
# Timing spikes-vs-calcium comparison. Runs the R.8b multi-lag timing analysis
# TWICE on the same cached spikes at N=1250: once on the raw binned SPIKES, once
# on the deconvolved CALCIUM feed. Cheap (N=1250); the blur effect is
# size-independent so 1250 makes the point.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r8b_signal.slurm

#SBATCH --job-name=ca_r8bsig
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/r8bsig_%j.log

set -eo pipefail
WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"
export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "host: $(hostname)   git: $(git rev-parse HEAD)   date: $(date)"
OUT="$WS/results/r8b"

for SIG in spikes feed; do
    echo "=== signal: $SIG ==="
    "$PY" scripts/r8b_multilag.py --net n1250_r4 --cache-dir "$WS/gt_cache" \
          --checkpoint-ms 500000 --chunk-ms 20000 --signal "$SIG" \
          --device cpu --out-dir "$OUT"
done

echo "done: $(date)"
ls -lh "$OUT"/r8b_n1250_r4_*.npz 2>/dev/null || echo "(no npz)"
echo
echo "rsync -avhz ${USER}@login1.nemo.uni-freiburg.de:$OUT/ ~/calcium_results/r8b/"
