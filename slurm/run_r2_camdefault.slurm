#!/bin/bash
# R.2 variant: dye-tau sweep with the camera held at the IDEALISED default
# (0.1 ms, the "infinitely fast" camera used everywhere else in this thesis)
# instead of run_r2.slurm's realistic 33 ms camera. Same compute script,
# same n1250_r4/T=500000 regime as the updated run_r2.slurm (see its comment
# for why) -- just one CLI flag different.
#
# This exact combination (n1250ai, T=100k) was tried repeatedly on a local
# laptop (11 GB RAM) and OOM-killed every time, once even from a genuine
# ~9.5 GB spike in the process itself that has no clear explanation -- do not
# retry locally, this belongs on the cluster.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r2_camdefault.slurm
#
# Plotting (fig_r2_plot.py) runs LOCALLY after rsyncing the npz back.

#SBATCH --job-name=ca_r2_r4_camdefault
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/r2_r4_camdefault_%j.log

set -eo pipefail
WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"
export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "host: $(hostname)   git: $(git rev-parse HEAD)   date: $(date)"
OUT="$WS/results/fig_data"
mkdir -p "$OUT"

"$PY" scripts/fig_r2_compute.py --net n1250_r4 --T-ms 500000 \
      --fixed-cam-ms 0.1 \
      --out "$OUT/r2_data_r4_camdefault.npz"

echo "done: $(date)"
ls -lh "$OUT/r2_data_r4_camdefault.npz" 2>/dev/null || echo "(no npz)"
echo
echo "rsync -avhz ${USER}@login1.nemo.uni-freiburg.de:$OUT/r2_data_r4_camdefault.npz ~/calcium_results/fig_data/"
