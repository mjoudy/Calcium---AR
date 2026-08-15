#!/bin/bash
# Linear Hawkes-process ground truth, N=1250, matched to LIF's adj_true
# (best_moments/n1250r4) -- the professor's suggested control for
# fig_linearity_way2 (Pernice et al. 2011's own model class, real spikes,
# unlike the OU arm). Two stages in one job so the RAM-heavy calcium+
# deconvolution step (confirmed >4GB and still climbing after 38s / ~10% of
# the run on an 11GB laptop -- killed rather than risk it) runs where there is
# headroom, not on someone's laptop next to other work:
#   1. scripts/hawkes_ground_truth.py   -- simulate spike events (~1 min, light)
#   2. scripts/hawkes_to_moments.py     -- calcium -> deconvolution -> Cxx/Cyx
#      (same pipeline as LIF/PIF; this is the part that needs the cluster)
#
# Needs adj_true.npy already present at $WS/best_moments/n1250r4/adj_true.npy
# -- rsync it up first (6.25 MB, only the topology, no spike data):
#   rsync -av ~/calcium_results/best_moments/n1250r4/adj_true.npy \
#     <cluster>:$(ws_find calcium_ar)/best_moments/n1250r4/
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_hawkes_ground_truth.slurm
#
# Then, locally:
#   rsync -av <cluster>:$(ws_find calcium_ar)/results/hawkes_moments/n1250_calcium/ \
#     ~/calcium_results/hawkes_moments/n1250_calcium/
#   python scripts/fig_linearity_way2.py   # once wired up as a 4th source

#SBATCH --job-name=ca_hawkes
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/hawkes_%j.log

set -eo pipefail

WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"

export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

GIT_COMMIT=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then GIT_DIRTY=yes; else GIT_DIRTY=no; fi
echo "host: $(hostname)   git: $GIT_COMMIT (dirty: $GIT_DIRTY)   date: $(date)"

ADJ="$WS/best_moments/n1250r4/adj_true.npy"
if [ ! -f "$ADJ" ]; then
  echo "missing $ADJ -- rsync it up first (see header comment)"; exit 1
fi

"$PY" scripts/hawkes_ground_truth.py \
  --adj "$ADJ" --out "$WS/hawkes_events/n1250" --sim-time 100000 --seed 1

"$PY" scripts/hawkes_to_moments.py \
  --events-dir "$WS/hawkes_events/n1250" \
  --out "$WS/results/hawkes_moments/n1250_calcium" \
  --chunk-ms 2000 --seed 1

echo "done: $(date)"
