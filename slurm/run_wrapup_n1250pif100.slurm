#!/bin/bash
# PIF ladder, tau_m x100 (20ms -> 2000ms), production pilot. eta=80.0 picked
# from scripts/pif_tau_probe.py --stage x100 (2026-08-15, properly-stationary
# probe, see the NETS["n1250_pif100"] comment in wrapup_run.py for the full
# eta/rate/CV table and the CV=6.14 caveat -- MUCH more irregular than the x10
# pilot, worth remembering when this feeds back into fig_linearity_way2).
#
# Uses wrapup_run_stream.py (not plain wrapup_run.py) with --save-moments so
# Cxx/Cyx land on disk directly -- same as how the x10 pilot
# (results/wrapup_n1250pif_T100k/seed1) was produced, and what
# fig_linearity_way2.py's DEFAULT_SOURCES expects.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_wrapup_n1250pif100.slurm
#
# Then, from your LOCAL machine:
#   rsync -avhz fr_mj200@login1.nemo.uni-freiburg.de:'/work/classic/fr_mj200-calcium_ar/results/wrapup_n1250pif100_T100k/' \
#     ~/calcium_results/wrapup_n1250pif100_T100k/
# and point fig_linearity_way2.py's "PIF pilot (tau_m x100)" source at
# ~/calcium_results/wrapup_n1250pif100_T100k/seed1 (not yet wired up -- ask
# Claude to add it once this data exists).

#SBATCH --job-name=ca_pif100
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/wrapup_pif100_%j.log

set -eo pipefail

WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"

export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

GIT_COMMIT=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then GIT_DIRTY=yes; else GIT_DIRTY=no; fi
echo "host: $(hostname)   git: $GIT_COMMIT (dirty: $GIT_DIRTY)   date: $(date)"

"$PY" scripts/wrapup_run_stream.py --net n1250_pif100 --sweep 100000 --seeds 1 \
  --save-moments

echo "done: $(date)"
