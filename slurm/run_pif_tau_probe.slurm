#!/bin/bash
# Eta probe for one stage of the PIF tau_m ladder (professor's "push tau_m to
# 10000x" comment). Run x100 first, look at the printed rate/CV table, THEN
# decide sim_time for x1000/x10000 (pif_tau_probe.py warns in its own output
# if sim_time looks too short relative to tau_m -- read that before trusting
# the eta pick for the larger stages).
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_pif_tau_probe.slurm x100
#          sbatch slurm/run_pif_tau_probe.slurm x1000
#          sbatch slurm/run_pif_tau_probe.slurm x10000    # read the caution in
#                                                          # pif_tau_probe.py first;
#                                                          # may need --sim-time raised
#                                                          # (and --time here raised to match)

#SBATCH --job-name=ca_pif_probe
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=slurm_logs/pif_tau_probe_%j.log

set -eo pipefail

STAGE="${1:?usage: sbatch run_pif_tau_probe.slurm <x100|x1000|x10000>}"

WS=$(ws_find calcium_ar)
PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"

export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

GIT_COMMIT=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then GIT_DIRTY=yes; else GIT_DIRTY=no; fi
echo "host: $(hostname)   git: $GIT_COMMIT (dirty: $GIT_DIRTY)   stage: $STAGE   date: $(date)"

"$PY" scripts/pif_tau_probe.py --stage "$STAGE"

echo "done: $(date)"
