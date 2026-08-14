#!/bin/bash
# R.4 scaling ladder, N=12500 20M-ms extension: FINAL aggregation step.
#
# Run this ONLY after both the pilot (seed 1) and the array job (seeds 2-5)
# have finished -- it scores every seed*/ dir found under each checkpoint and
# writes/overwrites metrics.csv. Cheap (CPU-only, reads cached moments), no
# GPU needed, but kept as its own SLURM job for logging consistency.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r4_n12500_extend20m_analyze.slurm

#SBATCH --job-name=ca_r4_n12500x20m_analyze
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/r4_n12500x20m_analyze_%j.log

set -eo pipefail

WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"

export CALCIUM_AR_WORKDIR="$WS"

echo "host: $(hostname)   git: $(git rev-parse HEAD)   date: $(date)"

for Tk in 100 200 500 1000 2000 5000 7500 10000 15000 20000; do
    d="$WS/results/wrapup_n12500r4_T${Tk}k"
    n=$(ls -d "$d"/seed*/ 2>/dev/null | wc -l)
    echo "=== T=${Tk}k : $n seed dirs found ==="
    "$PY" scripts/analyze_run.py --data "$d" || echo "  analyze T${Tk}k failed (continuing)"
done

echo "done: $(date)"
echo "--- copy home just these: ---"
ls -lh "$WS"/results/wrapup_n12500r4_T*k/metrics.csv 2>/dev/null
