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
# Needs an adj_true.npy for the n1250_r4 network. The local copy at
# ~/calcium_results/best_moments/n1250r4/adj_true.npy was itself almost
# certainly rsynced DOWN from an earlier n1250_r4 cluster run (wrapup_run.py
# or wrapup_run_stream.py, --net n1250_r4) -- so the original is probably
# still sitting on the cluster and there is no need to push a local copy back
# up. This job searches a few likely locations for it (below) before giving
# up; if none match, either point ADJ_OVERRIDE at the right file when
# submitting (`sbatch --export=ADJ_OVERRIDE=/path/to/adj_true.npy ...`) or,
# only as a last resort, rsync the local copy up. NOTE: $(ws_find ...) only
# resolves on the cluster -- from your LOCAL machine use the literal
# workspace path (this project's is /work/classic/fr_mj200-calcium_ar), not
# $(ws_find calcium_ar), which would try (and fail) to run locally:
#   rsync -avhz ~/calcium_results/best_moments/n1250r4/adj_true.npy \
#     fr_mj200@login1.nemo.uni-freiburg.de:/work/classic/fr_mj200-calcium_ar/best_moments/n1250r4/
#
# Submit (on the cluster, where $(ws_find ...) does resolve):
#          cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_hawkes_ground_truth.slurm
#
# Then, from your LOCAL machine:
#   rsync -avhz fr_mj200@login1.nemo.uni-freiburg.de:'/work/classic/fr_mj200-calcium_ar/results/hawkes_moments/n1250_calcium_tuned/' \
#     ~/calcium_results/hawkes_moments/n1250_calcium_tuned/
#   python scripts/fig_linearity_way2.py
#
# RECALIBRATED 2026-08-15: the first version (rate0=10Hz default, margin=0.3)
# came back with AUC=0.54 (barely above chance) -- the analytic zero-noise
# check showed margin=0.3 already has the BEST relative true/non-edge
# separation of the margins tried (stronger coupling dilutes it further, more
# indirect-path contribution per Pernice's own math), so the fix is not more
# coupling -- it's rate. Hawkes realizes only ~6.1Hz at rate0=10Hz vs LIF's
# ~14.8Hz, i.e. much less averaging for the SAME calcium noise floor.
# Realized rate scales linearly with rate0 (y=(1-G)^-1 y0), so rate0=24.3
# lands at ~14.2-14.3Hz -- verified locally (both stages): mean rate 14.23Hz,
# matches the closed-form Eq.10 prediction (14.07/14.20 Hz E/I) closely.
# Old (untuned) result kept at .../hawkes_moments/n1250_calcium for reference
# -- this run writes to a separate _tuned path, does not overwrite it.

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

# Try, in order: an explicit override, a previously-staged copy, the plain
# wrapup_run.py output, the streaming wrapup_run_stream.py output (any T).
# First one that exists wins.
ADJ=""
for CAND in \
    "${ADJ_OVERRIDE:-}" \
    "$WS/best_moments/n1250r4/adj_true.npy" \
    "$WS/results/wrapup_n1250r4/seed1/adj_true.npy" \
    $WS/results/wrapup_n1250r4_T*/seed1/adj_true.npy
do
  if [ -n "$CAND" ] && [ -f "$CAND" ]; then ADJ="$CAND"; break; fi
done
if [ -z "$ADJ" ]; then
  echo "no adj_true.npy found for n1250_r4 under $WS -- searched:"
  echo "  \$ADJ_OVERRIDE, $WS/best_moments/n1250r4/, $WS/results/wrapup_n1250r4*/seed1/"
  echo "point ADJ_OVERRIDE at the right file, or rsync one up (see header comment)"
  exit 1
fi
echo "using adj_true.npy: $ADJ"

"$PY" scripts/hawkes_ground_truth.py \
  --adj "$ADJ" --out "$WS/hawkes_events/n1250_tuned" --sim-time 100000 \
  --rate0 24.3 --margin 0.30 --seed 1

"$PY" scripts/hawkes_to_moments.py \
  --events-dir "$WS/hawkes_events/n1250_tuned" \
  --out "$WS/results/hawkes_moments/n1250_calcium_tuned" \
  --seed 1
  # chunk-ms left at the script default (10s, matches wrapup_run_stream.py's
  # own default) -- an earlier run here passed --chunk-ms 2000 as an extra RAM
  # margin the 96G job never needed, and that starved the one-time first-chunk
  # tau estimate (came out -330ms instead of ~+100ms, silently broke the
  # deconvolution). 96G has plenty of headroom at the default chunk size.

echo "done: $(date)"
