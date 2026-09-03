#!/bin/bash
# Re-score cached wrap-up runs with the extended metric set (adds the MCC family:
# mcc3, mcc_pres, mcc_E, mcc_I, mcc_type, mcc_neuron) to each run's metrics.csv.
#
# CPU-only, no GPU, no re-simulation, no re-solving: it just reads the cached
# seed*/A_*.npy + adj_true.npy and rewrites metrics.csv (same existing columns,
# same values, six new columns). Needs the estimate matrices to still be on disk
# -- if a run dir was pruned to moments only, it is skipped with a warning.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_mcc_rescore.slurm

#SBATCH --job-name=ca_mcc_rescore
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/mcc_rescore_%j.log

set -eo pipefail

WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"
export CALCIUM_AR_WORKDIR="$WS"
mkdir -p slurm_logs

echo "host: $(hostname)   git: $(git rev-parse --short HEAD)   date: $(date)"

# Size ladder (tuned-AI 'r4' family: the one that spans every N) + the two other
# N=12500 dynamical regimes (canonical, low-rate). Edit the glob list to taste.
DIRS=(
    "$WS"/results/wrapup_n1250r4_T*k
    "$WS"/results/wrapup_n2500r4_T*k
    "$WS"/results/wrapup_n5000r4_T*k
    "$WS"/results/wrapup_n12500r4_T*k
    "$WS"/results/wrapup_n12500_T*k
    "$WS"/results/wrapup_n12500lr_T*k
)

scored=0; skipped=0
for d in "${DIRS[@]}"; do
    [ -d "$d" ] || continue
    nA=$(ls "$d"/seed*/A_*.npy 2>/dev/null | wc -l)
    if [ "$nA" -eq 0 ]; then
        echo "SKIP  $(basename "$d")  (no seed*/A_*.npy on disk -- pruned?)"
        skipped=$((skipped+1)); continue
    fi
    echo "=== $(basename "$d")  ($nA estimate matrices) ==="
    if "$PY" scripts/analyze_run.py --data "$d"; then
        scored=$((scored+1))
    else
        echo "  FAILED on $(basename "$d") (continuing)"; skipped=$((skipped+1))
    fi
done

echo
echo "done: $(date)   scored=$scored  skipped=$skipped"
echo "--- copy home just these: ---"
for d in "${DIRS[@]}"; do
    [ -f "$d/metrics.csv" ] && ls -lh "$d/metrics.csv"
done
