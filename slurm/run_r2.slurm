#!/bin/bash
# R.2 (calcium-observation section): one NEST simulation, then score OLS on
# five signal chains (spikes / raw+deconv x {dye tau sweep, camera rate
# sweep}), each swept ONE VARIABLE AT A TIME (see fig_r2_compute.py docstring)
# with ranges extended past the old endpoints so both curves reach their
# floor/plateau. CPU only, no NEST densify.
#
# Uses the n1250_r4 regime (V_reset=10, properly irregular CV~1 AI firing --
# the SAME network as fig_best_n1250r4 / fig_preproc_n1250 / Table 2) and
# T=500,000 ms (matching fig_preproc/fig_best, not a shorter diagnostic
# recording) so the absolute numbers here are directly comparable to the rest
# of the thesis, not just the relative shape of the curves. The original
# n1250ai/T=100k run undersold both deconvolution and the calcium pipeline
# generally (V_reset=0 -> too-regular firing, 5x less data).
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r2.slurm
#
# Plotting (fig_r2_plot.py) runs LOCALLY after rsyncing r2_data_r4.npz back.

#SBATCH --job-name=ca_r2_r4
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/r2_r4_%j.log

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
      --out "$OUT/r2_data_r4.npz"

echo "done: $(date)"
ls -lh "$OUT/r2_data_r4.npz" 2>/dev/null || echo "(no npz)"
echo
echo "rsync -avhz ${USER}@login1.nemo.uni-freiburg.de:$OUT/r2_data_r4.npz ~/calcium_results/fig_data/"
