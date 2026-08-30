#!/bin/bash
# R.2 follow-up: fixed realistic camera interval (33 ms), sweep recording
# length T to see whether more data helps and where it plateaus. CPU only
# (fig_r2_compute.py's pipeline is plain numpy/scipy, no GPU).
#
# Measured locally: ~65-70s/point at T=100k ms; cost scales with T (fine-
# resolution processing, not output size). Rough estimate for the full
# 5-point x 2-kind sweep (100k/500k/1M/2M/5M ms): ~3h. Generous budget below.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r2_longT.slurm

#SBATCH --job-name=ca_r2_longT
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/r2_longT_%j.log

set -eo pipefail

WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"

export CALCIUM_AR_WORKDIR="$WS"

echo "host: $(hostname)  git: $(git rev-parse HEAD)  date: $(date)"

"$PY" scripts/fig_r2_longT.py --net n1250_r4 --interval-ms 33 \
      --Ts 100000 500000 1000000 2000000 5000000 \
      --out "$WS/results/fig_R2_longT"

echo "done: $(date)"
echo "--- copy home just these: ---"
ls -lh "$WS"/results/fig_R2_longT* 2>/dev/null
echo
echo "rsync -avhz \${USER}@login1.nemo.uni-freiburg.de:$WS/results/fig_R2_longT* ~/calcium_results/"
