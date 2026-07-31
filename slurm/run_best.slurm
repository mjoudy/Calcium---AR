#!/bin/bash
# Best-results showcase — reduce GT + inferred matrices to small views (full
# block-averaged + a mixed exc/inh sub-block) for the flagship figure. Reads the
# already-saved A_ols.npy + adj_true.npy; only small npz travel home. No GPU.
#
#   flagship (best realistic) : N=12500 low-rate 14 Hz, 5M ms  (wrapup_n12500lr_T5000k)
#   companion small           : N=1250  r4 14 Hz, 500k ms      (wrapup_n1250r4_T500k)
#                               --full-error also classifies the FULL N=1250 matrix
#                               at full resolution (no block-averaging) for the
#                               (GT, inferred, error) x (full, sub-block) grid figure.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_best.slurm

#SBATCH --job-name=ca_best
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/best_%j.log

set -eo pipefail
WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"
export CALCIUM_AR_WORKDIR="$WS"

echo "host: $(hostname)   git: $(git rev-parse HEAD)   date: $(date)"
OUT="$WS/results/best"

"$PY" scripts/fig_best_compute.py --data "$WS/results/wrapup_n12500lr_T5000k/seed1" \
      --tag n12500lr --out-dir "$OUT"
"$PY" scripts/fig_best_compute.py --data "$WS/results/wrapup_n1250r4_T500k/seed1" \
      --tag n1250r4 --out-dir "$OUT" --full-error

echo "done: $(date)"
ls -lh "$OUT"/*.npz 2>/dev/null || echo "(no npz)"
echo
echo "rsync -avhz ${USER}@login1.nemo.uni-freiburg.de:$OUT/ ~/calcium_results/best/"
