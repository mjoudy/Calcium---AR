#!/bin/bash
# Pool the 7 chunks from run_n12500_kpool_test.slurm into one estimate and
# score it. Cheap (just loads 7x Cxx/Cyx, averages, one OLS solve) -- CPU is
# fine, minutes not hours.
#
# Submit AFTER the array job finishes (or chain with --dependency, see the
# submit commands): sbatch slurm/run_n12500_kpool_pool.slurm

#SBATCH --job-name=ca_n12500_kpool_solve
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/n12500_kpool_solve_%j.log

set -eo pipefail

WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"

export CALCIUM_AR_WORKDIR="$WS"

DIR="$WS/results/wrapup_n12500r4_T12000k"
echo "pooling chunks under: $DIR"
ls -d "$DIR"/seed2* 2>/dev/null || { echo "FATAL: no seed20x dirs found yet"; exit 1; }

"$PY" scripts/pool_chunks_and_solve.py --dir "$DIR" --out "$DIR"_pooled

echo "done: $(date)"
echo "--- copy home just this: ---"
echo "rsync -avhz \${USER}@login1.nemo.uni-freiburg.de:${DIR}_pooled/ ~/calcium_results/n12500_kpool_pooled/"
