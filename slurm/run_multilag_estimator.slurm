#!/bin/bash
# Joint multi-lag estimator vs the existing single-lag baseline: does
# conditioning on several past lags AT ONCE (not one at a time, unlike the
# R.8b diagnostic) reduce false-positive strength? Runs on cached
# ground-truth spikes (gt_cache/n1250_r4_seed1_*.npz), both on raw spikes
# (sanity check) and the deconvolved calcium feed (the real test). CPU only
# -- N=1250 inverses of a (4*1250)x(4*1250) matrix are still fast, no GPU
# queue wait needed.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_multilag_estimator.slurm

#SBATCH --job-name=ca_multilag
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/multilag_%j.log

set -eo pipefail

WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"
export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "host: $(hostname)   git: $(git rev-parse HEAD)   date: $(date)"
OUT="$WS/results/multilag"

"$PY" scripts/multilag_estimator.py --net n1250_r4 --signal spikes \
    --lags-ms 1.5 3.0 4.5 6.0 --device cpu --out-dir "$OUT"

"$PY" scripts/multilag_estimator.py --net n1250_r4 --signal feed \
    --lags-ms 1.5 3.0 4.5 6.0 --device cpu --out-dir "$OUT"

echo "done: $(date)"
ls -lh "$OUT"/*.npz 2>/dev/null || echo "(no npz)"
echo
echo "rsync -avhz ${USER}@login1.nemo.uni-freiburg.de:$OUT/ ~/calcium_results/multilag/"
