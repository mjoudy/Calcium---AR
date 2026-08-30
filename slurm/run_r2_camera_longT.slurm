#!/bin/bash
# R.2, ONE longer recording length: regenerate just the camera-rate panel
# (like fig_R2's top-left) at T=1,000,000 ms instead of 500,000 ms, to check
# whether the realistic ~33ms-camera point's ROC-AUC improves with more data.
# Skips the tau sweep and uses a focused frame list (not the full 13-point
# default) -- keeps this to hours, not most of a day.
#
# Measured locally: ~65-70s/point at T=100k ms, scales ~linearly with T ->
# ~700s (~12 min) per point at T=1M ms. 4 frames x 2 kinds (deconv/raw) = 8
# calcium points ~= 1.6h, + 7 cheap spike-bin points (no calcium, seconds
# each). Budget below has margin.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r2_camera_longT.slurm

#SBATCH --job-name=ca_r2_cam_longT
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/r2_cam_longT_%j.log

set -eo pipefail

WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"

export CALCIUM_AR_WORKDIR="$WS"

echo "host: $(hostname)  git: $(git rev-parse HEAD)  date: $(date)"

OUT="$WS/results/fig_data/r2_camera_longT.npz"
mkdir -p "$(dirname "$OUT")"

"$PY" scripts/fig_r2_compute.py --net n1250_r4 --T-ms 1000000 \
      --kinds spikes deconv_rate raw_rate \
      --frames 10 20 33 50 \
      --out "$OUT"

echo "done: $(date)"
echo "rsync -avhz \${USER}@login1.nemo.uni-freiburg.de:$OUT ~/calcium_results/fig_data/"
