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
#          sbatch slurm/run_n12500_lowrate_gpu.slurm

#SBATCH --job-name=ca_n12500_lowrate
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/n12500_lowrate_%j.log
#
# Measured: the 1M-ms probe took 1h14m. The 5M sweep is ~5x that work, and the
# observed pace suggests ~6-8 h, so the wall is 12 h. Checkpoints are solved and
# SAVED as soon as they are reached, so even a timeout leaves the shorter
# recording lengths on disk. Scoring runs on the cluster (metrics.csv only).

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

# --- preflight: what regime does the CANONICAL config actually produce here? ---
# Cheap (short sim), and it lands in the log before the expensive part starts, so
# a bad regime can be caught early.
echo "=== regime preflight: canonical Brunel at N=12500 ==="
mkdir -p "$WS/results/network_state"
"$PY" scripts/regime_probe.py --scale n12500 --only brunel_AI_lowrate \
      --sim-time 5000 --n-threads "${SLURM_CPUS_PER_TASK}" \
      --fig "$WS/results/network_state/n12500_AI_lowrate.png" || true

# A2 — recovery curve at full scale, CANONICAL Brunel 2000 parameters
# (J=0.1, g=5, eta=2, V_reset=10 mV at N=12500 — no rescaling needed at this size).
# ONE 5M-ms simulation per seed, scored at the 1M / 2M / 5M checkpoints, each
# solved and SAVED as it is reached. 1 seed (seeds deferred to a revision).
"$PY" scripts/wrapup_run_stream.py --net n12500_lowrate \
      --sweep 1000000 2000000 5000000 --seeds 1 \
      --methods ols ridge --device cuda --chunk-ms 5000

# Score ON THE CLUSTER — only the small metrics.csv needs to travel home.
for Tk in 1000 2000 5000; do
    "$PY" scripts/analyze_run.py --data "$WS/results/wrapup_n12500lr_T${Tk}k" \
        || echo "analyze T${Tk}k failed (continuing)"
done
echo "done: $(date)"
echo "--- copy home just these: ---"
ls -lh "$WS"/results/wrapup_n12500lr_T*k/metrics.csv "$WS"/results/network_state/*.png 2>/dev/null
