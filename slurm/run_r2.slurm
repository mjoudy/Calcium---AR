#!/bin/bash
# R.2 (calcium-observation section): one NEST simulation (N=1250), then score
# OLS on five signal chains (spikes / raw+deconv x {dye tau sweep, camera rate
# sweep}), each swept ONE VARIABLE AT A TIME (see fig_r2_compute.py docstring)
# with ranges extended past the old endpoints so both curves reach their
# floor/plateau. CPU only, no NEST densify — cheap at N=1250.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r2.slurm
#
# Plotting (fig_r2_plot.py) runs LOCALLY after rsyncing r2_data.npz back.

#SBATCH --job-name=ca_r2
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/r2_%j.log

set -eo pipefail
WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"
export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "host: $(hostname)   git: $(git rev-parse HEAD)   date: $(date)"
OUT="$WS/results/fig_data"
mkdir -p "$OUT"

"$PY" scripts/fig_r2_compute.py --net n1250ai --T-ms 100000 \
      --out "$OUT/r2_data.npz"

echo "done: $(date)"
ls -lh "$OUT/r2_data.npz" 2>/dev/null || echo "(no npz)"
echo
echo "rsync -avhz ${USER}@login1.nemo.uni-freiburg.de:$OUT/r2_data.npz ~/calcium_results/fig_data/"
