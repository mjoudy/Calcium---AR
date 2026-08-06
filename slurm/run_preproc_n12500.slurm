#!/bin/bash
# Effect of preprocessing at FULL SCALE (N=12500): does the deconvolved-feed
# recovery benefit seen at N=1250 (fig_preproc_n1250) survive when synapses are
# realistically weaker for a bigger network?
#
# n12500_r4 is the R.4-ladder counterpart of n1250_r4: same dynamical regime
# (g=8, matched ~14 Hz / CV~1 AI state), but J_ex fluctuation-scaled down from
# 0.474 to 0.150 as N grows from 1250 to 12500 (see wrapup_run.py NETS). Same
# recording length ratio too: checkpoint-ms=5,000,000 keeps T/N=400, identical
# to the 500,000 ms used for n1250_r4 in run_preproc.slurm.
#
# Cache-aware: if gt_cache/n12500_r4_seed1_*.npz is missing this job simulates
# it first (same cost as run_r4_n12500.slurm's cache-build step, ~12h). If the
# cache already exists (e.g. run_r4_n12500.slurm already ran), that step is
# skipped and this job just does the two streaming passes (raw vs deconvolved).
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_preproc_n12500.slurm

#SBATCH --job-name=ca_preproc_n12500
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=16:00:00
#SBATCH --output=slurm_logs/preproc_n12500_%j.log
#
# 240G / 16h: worst case (cache missing) = one 5,000,000 ms N=12500 simulation
# (~875M spike events, same budget as run_r4_n12500.slurm's ~12h) + two
# streaming raw-vs-deconvolved passes over it. If the cache already exists,
# only the two streaming passes run and this finishes in a couple of hours.

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
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available(), "No CUDA torch — set GPU_PY to a CUDA env."
print("gpu:", torch.cuda.get_device_name(0))
PYCHK

CACHE="$WS/gt_cache"
mkdir -p "$CACHE"

if ! ls "$CACHE"/n12500_r4_seed1_*.npz >/dev/null 2>&1; then
    echo "=== no cached n12500_r4 spikes found — simulating first (long) ==="
    "$PY" scripts/wrapup_run_stream.py --net n12500_r4 \
          --sweep 5000000 --seeds 1 --methods ols --device cuda \
          --chunk-ms 5000 --cache-dir "$CACHE"
else
    echo "=== found cached n12500_r4 spikes, skipping simulation ==="
    ls -lh "$CACHE"/n12500_r4_seed1_*.npz
fi

OUT="$WS/results/preproc"
"$PY" scripts/fig_preproc_compute.py --net n12500_r4 --cache-dir "$CACHE" \
      --checkpoint-ms 5000000 --device cuda --out-dir "$OUT"

echo "done: $(date)"
ls -lh "$OUT"/*.npz 2>/dev/null || echo "(no npz)"
echo
echo "rsync -avhz \${USER}@login1.nemo.uni-freiburg.de:$OUT/ ~/calcium_results/preproc/"
