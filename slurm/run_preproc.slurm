#!/bin/bash
# Effect of preprocessing (refresh at scale): OLS on raw calcium vs on the
# deconvolved feed, same cached spikes and same calcium, scored with the R.1-R.8
# scorer. Two streaming passes from cached spikes (no NEST).
#
#   N=1250 : cpu, quick.   (Add n12500_r4 on l40s if you want it at full scale.)
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_preproc.slurm

#SBATCH --job-name=ca_preproc
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/preproc_%j.log

set -eo pipefail
WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"
export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "host: $(hostname)   git: $(git rev-parse HEAD)   date: $(date)"
OUT="$WS/results/preproc"

"$PY" scripts/fig_preproc_compute.py --net n1250_r4 --cache-dir "$WS/gt_cache" \
      --checkpoint-ms 500000 --device cpu --out-dir "$OUT"

echo "done: $(date)"
ls -lh "$OUT"/*.npz 2>/dev/null || echo "(no npz)"
echo
echo "rsync -avhz ${USER}@login1.nemo.uni-freiburg.de:$OUT/ ~/calcium_results/preproc/"
