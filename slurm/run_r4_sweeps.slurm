#!/bin/bash
# R.4 scaling sweep: the same AI regime (~14 Hz, CV ~0.63) at four network sizes,
# each scored at MATCHED samples-per-neuron T/N = 800 / 1600 / 4000.
#
#   N= 1250 (J=0.316, eta=1.05):  100k /  200k /  500k ms
#   N= 2500 (J=0.224, eta=1.10):  200k /  400k / 1000k ms
#   N= 5000 (J=0.158, eta=1.30):  400k /  800k / 2000k ms
#   N=12500 (J=0.100, eta=1.50): 1000k / 2000k / 5000k ms  <- ALREADY DONE
#                                 (results/wrapup_n12500lr_T*k, reused)
#
# OLS only (the estimator story lives in R.5). Moments and ground truth are
# cached so any later estimator experiment at these sizes is seconds, not hours.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r4_sweeps.slurm

#SBATCH --job-name=ca_r4_sweeps
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/r4_sweeps_%j.log

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

run () {   # $1 = preset, $2.. = recording lengths (ms)
    local net="$1"; shift
    echo "=== $net : $* ==="
    "$PY" scripts/wrapup_run_stream.py --net "$net" --sweep "$@" --seeds 1 \
          --methods ols --device cuda --chunk-ms 5000 \
          --cache-dir "$WS/gt_cache" --save-moments
}

run n1250_r4  100000  200000   500000
run n2500_r4  200000  400000  1000000
run n5000_r4  400000  800000  2000000

# score on the cluster; only the small metrics.csv files travel home
for d in "$WS"/results/wrapup_n{1250,2500,5000}r4_T*k; do
    [ -d "$d" ] && "$PY" scripts/analyze_run.py --data "$d" \
        || echo "analyze $d failed (continuing)"
done

echo "done: $(date)"
echo "--- copy home just these: ---"
ls -lh "$WS"/results/wrapup_n{1250,2500,5000}r4_T*k/metrics.csv 2>/dev/null
