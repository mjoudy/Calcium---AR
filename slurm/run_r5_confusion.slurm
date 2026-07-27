#!/bin/bash
# R.5 confusion matrices for N=12500 (GPU). Re-solves the cached moments at the
# best-lambda operating point for OLS / Lasso / Lasso+Dale, at three recording
# lengths, and builds the 3x3 confusion. Short — only 3 methods x 3 T/N.
#
# N=1250 does NOT need this job: run it locally with
#   python scripts/fig_r5_confusion.py --root <moments> --net n1250r4 \
#       --csv results/r5/r5_n1250r4.csv --device cpu
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r5_confusion.slurm

#SBATCH --job-name=ca_r5conf
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/r5conf_%j.log

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

R5OUT="$WS/results/r5"
# N=1250 too (cpu path, fast) so both npz are produced in one job
"$PY" scripts/fig_r5_confusion.py --root "$WS/results" --net n1250r4 \
      --csv "$R5OUT/r5_n1250r4.csv" --methods ols lasso lassodale \
      --device cpu --out-dir "$R5OUT"
"$PY" scripts/fig_r5_confusion.py --root "$WS/results" --net n12500r4 \
      --csv "$R5OUT/r5_n12500r4.csv" --methods ols lasso lassodale \
      --device cuda --out-dir "$R5OUT"

echo "done: $(date)"
ls -lh "$R5OUT"/r5conf_*.npz 2>/dev/null || echo "(no npz)"
echo
echo "rsync -avhz ${USER}@login1.nemo.uni-freiburg.de:$R5OUT/r5conf_*.npz ~/calcium_results/r5/"
