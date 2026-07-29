#!/bin/bash
# R.8 (shared input vs directionality) core — both sizes. Reads the already-saved
# A_ols.npy + adj_true.npy at the longest recording and measures the directional
# asymmetry of predicted edges (true vs false positive), then the precision/recall
# gained by filtering symmetric edges. No simulation, no solve, no GPU — just load
# and sample. Fast.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r8.slurm

#SBATCH --job-name=ca_r8
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/r8_%j.log

set -eo pipefail

WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"
export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "host: $(hostname)   git: $(git rev-parse HEAD)   date: $(date)"
R8OUT="$WS/results/r8"

"$PY" scripts/fig_r8_compute.py --net n1250r4  --out-dir "$R8OUT"
"$PY" scripts/fig_r8_compute.py --net n12500r4 --out-dir "$R8OUT"

echo "done: $(date)"
ls -lh "$R8OUT"/*.npz 2>/dev/null || echo "(no npz)"
echo
echo "rsync -avhz ${USER}@login1.nemo.uni-freiburg.de:$R8OUT/ ~/calcium_results/r8/"
