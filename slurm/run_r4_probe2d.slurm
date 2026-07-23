#!/bin/bash
# 2-D regime probe at N=1250: can we get a realistic rate AND textbook-AI
# irregularity (CV ~ 1) at the same time?
#
# The R.4 ladder currently sits at ~14 Hz but CV ~0.63 — asynchronous, only
# moderately irregular. Dropping eta to get a realistic rate also costs CV.
# g pushes CV back up, and J is a third knob if g alone is not enough.
#
#   pass 1: J = 0.316 (fluctuation-scaled, the ladder's J)   6 g x 8 eta
#   pass 2: J x 1.5
#   pass 3: J x 2.0
#
# ~144 short simulations at N=1250, statistics from sparse events only.
# Every combination is written to results/regime2d/*.csv (+ .npz for figures).
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r4_probe2d.slurm

#SBATCH --job-name=ca_probe2d
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/probe2d_%j.log

set -eo pipefail

WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"

export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "host: $(hostname)   git: $(git rev-parse HEAD)   date: $(date)"

OUT="$WS/results/regime2d"

for JS in 1.0 1.5 2.0; do
    echo; echo "############ J scale x${JS} ############"
    "$PY" scripts/r4_probe_2d.py --sizes 1250 --j-scale "$JS" \
          --sim-time 4000 --n-threads "${SLURM_CPUS_PER_TASK}" \
          --target-rate 14 --target-cv 1.0 --out-dir "$OUT"
done

echo; echo "done: $(date)"
echo "--- copy home (tiny files) ---"
ls -lh "$OUT"
echo
echo "rsync -av nemo:$OUT/ ~/calcium_results/regime2d/"
