#!/bin/bash
# R.4 gating step: tune eta per network size so every size sits in the SAME
# regime (AI) at the SAME rate (~14 Hz), leaving N as the only variable in the
# scaling sweep.
#
# Cost: 3 sizes x 14 eta values = 42 short (4 s) simulations. N=12500 is SKIPPED
# because it is already tuned (the existing low-rate run: g=6, eta=1.5, J=0.1,
# V_reset=10 -> 13.7 Hz, CV 0.62) and it was also the slowest to simulate.
# Statistics come from sparse spike events, so memory stays modest.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r4_tune.slurm
#
# Then read the "=== R.4 configs ===" block at the end of the log — those eta/J
# values define the sweep that follows.

#SBATCH --job-name=ca_r4_tune
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/r4_tune_%j.log

set -eo pipefail

WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"

export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "host: $(hostname)   git: $(git rev-parse HEAD)   date: $(date)"

"$PY" scripts/r4_tune_regime.py --target 14 --sim-time 4000 \
      --n-threads "${SLURM_CPUS_PER_TASK}"

echo "done: $(date)"
echo "--- copy the '=== R.4 configs ===' block above ---"
