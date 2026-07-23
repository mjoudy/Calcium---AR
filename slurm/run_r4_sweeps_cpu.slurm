#!/bin/bash
# R.4 sweeps for the three SMALL sizes on the cpu partition — same science as
# slurm/run_r4_sweeps.slurm, no GPU. Use this when the l40s nodes are drained.
#
# At N<=5000 the numpy path is adequate: the moment accumulation is the only
# heavy part and the matrices are at most 5000^2. Slower than the L40S, but it
# runs now instead of waiting for the GPU partition to come back.
#
# N=12500 is NOT here — that one really wants the GPU (slurm/run_r4_n12500.slurm).
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r4_sweeps_cpu.slurm
#
# If you run this, CANCEL the queued GPU twin (scancel 7587301) so the two do not
# write the same result directories.

#SBATCH --job-name=ca_r4_sweeps_cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/r4_sweeps_cpu_%j.log

set -eo pipefail

WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"

export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "host: $(hostname)   git: $(git rev-parse HEAD)   date: $(date)"

run () {   # $1 = preset, $2.. = recording lengths (ms)
    local net="$1"; shift
    echo "=== $net : $*   ($(date)) ==="
    "$PY" scripts/wrapup_run_stream.py --net "$net" --sweep "$@" --seeds 1 \
          --methods ols --chunk-ms 5000 \
          --cache-dir "$WS/gt_cache" --save-moments
}

# smallest first: if the wall runs out, the cheap rungs are already on disk
run n1250_r4  100000  200000   500000
run n2500_r4  200000  400000  1000000
run n5000_r4  400000  800000  2000000

for d in "$WS"/results/wrapup_n{1250,2500,5000}r4_T*k; do
    [ -d "$d" ] && "$PY" scripts/analyze_run.py --data "$d" \
        || echo "analyze $d failed (continuing)"
done

echo "done: $(date)"
ls -lh "$WS"/results/wrapup_n{1250,2500,5000}r4_T*k/metrics.csv 2>/dev/null
