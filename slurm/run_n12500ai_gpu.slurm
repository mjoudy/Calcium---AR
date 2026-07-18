#!/bin/bash
# Full Brunel scale (N=12500) in the clean-AI regime, GPU-accelerated.
# The heavy lag-pair moment accumulation runs on the L40S GPU (float32 matmuls);
# the OLS/Ridge solve is one exact CPU inverse. Longer recordings are needed at
# this scale (quality tracks samples/parameter = T/N, and N is 10x larger), so we
# sweep 1M / 2M / 5M ms.
#
# REQUIRES a CUDA-enabled torch env. Set GPU_PY to its python. If you don't have
# one yet, either create it, or run the CPU fallback:
#   sbatch --partition=cpu --gres=  ... (edit) OR run wrapup_run_stream.py WITHOUT
#   --device (numpy path) on the cpu partition at a smaller T (e.g. 1M ms, ~1 h).
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_n12500ai_gpu.slurm

#SBATCH --job-name=ca_n12500ai_gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/n12500ai_gpu_%j.log
#
# PROBE first (1 seed, 1M ms) to measure real cost now that deconvolution is on
# the GPU. Read the log for the per-checkpoint time, then scale up by editing the
# --sweep / --seeds line below, e.g.:
#   --sweep 1000000 2000000 5000000 --seeds 1 2   (and bump --time accordingly)

set -eo pipefail

WS=$(ws_find calcium_ar)
# CUDA-enabled python: override with GPU_PY if your torch+CUDA env lives elsewhere.
PY="${GPU_PY:-$WS/env-gpu/bin/python}"
[ -x "$PY" ] || PY="$WS/env/bin/python"      # fall back to the default env
cd "$WS/Calcium---AR"

export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "host: $(hostname)   git: $(git rev-parse HEAD)   date: $(date)   PY=$PY"
"$PY" - <<'PYCHK'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available(), "No CUDA torch — set GPU_PY to a CUDA env, or use the CPU fallback (see header)."
print("gpu:", torch.cuda.get_device_name(0))
PYCHK

# 2 seeds x {1M,2M,5M} ms, OLS+Ridge only, GPU accumulation
"$PY" scripts/wrapup_run_stream.py --net n12500ai \
      --sweep 1000000 2000000 5000000 --seeds 1 2 \
      --methods ols ridge --device cuda --chunk-ms 5000
echo "done: $(date)"
