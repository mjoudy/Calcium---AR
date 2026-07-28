#!/bin/bash
# R.7 (hidden neurons / shared input) at N=12500. Re-solves the cached moments at
# four observation fractions (1.0 / 0.5 / 0.25 / 0.1), holding samples-per-observed
# neuron ~constant by pairing each fraction with a matched-length checkpoint:
#
#   f=1.0  |S|=12500  T=1000k   (T/|S| ~ 800)
#   f=0.5  |S|= 6250  T= 500k
#   f=0.25 |S|= 3125  T= 250k
#   f=0.1  |S|= 1250  T= 125k
#
# so any drop is HIDDEN-INPUT confounding, not less data. Plus the shared-input
# mechanism panel at f=0.5. No new simulation. GPU for the inverses (up to 12500).
#
# NOTE needs the R.5 short-T moments (T125k/250k/500k) — produced by run_r5.slurm
# phase 1. If those dirs lack Cxx.npy, run run_r5.slurm first.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r7.slurm

#SBATCH --job-name=ca_r7
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/r7_%j.log

set -eo pipefail

WS=$(ws_find calcium_ar)
PY="${GPU_PY:-$WS/env-gpu/bin/python}"
[ -x "$PY" ] || PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"

export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "host: $(hostname)   git: $(git rev-parse HEAD)   date: $(date)   PY=$PY"
"$PY" - <<'PYCHK'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
assert torch.cuda.is_available(), "no CUDA torch — set GPU_PY to a CUDA env."
PYCHK

R7OUT="$WS/results/r7"
"$PY" scripts/fig_r7_compute.py --root "$WS/results" --net n12500r4 \
      --fractions 1.0 0.5 0.25 0.1 --base-tn 800 --shared-frac 0.5 \
      --device cuda --out-dir "$R7OUT"

echo "done: $(date)"
ls -lh "$R7OUT"/* 2>/dev/null || echo "(no output)"
echo
echo "rsync -avhz ${USER}@login1.nemo.uni-freiburg.de:$R7OUT/ ~/calcium_results/r7/"
