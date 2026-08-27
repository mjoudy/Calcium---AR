#!/bin/bash
# ONE-OFF TEST: how far can N=12500 go with K nodes pooling independent noise
# realizations of the SAME network, instead of one node holding the whole
# recording in memory? Single test, no seed replicates -- just "how far".
#
# Mechanism: each array task builds the IDENTICAL connectivity (loaded from
# an existing seed1 adj_true.npy -- adjacency_override in BrunelNetwork, see
# calcium_ar/simulation/brunel_network.py) but simulates its own T/K ms with
# its own noise seed (200 + array index). Each task saves its own normalized
# Cxx.npy/Cyx.npy (--save-moments). Once all K finish, pool with
# scripts/pool_chunks_and_solve.py (equal-length chunks -> plain average of
# the saved Cxx/Cyx is exact) and solve OLS once.
#
# K=7, chunk=12,000,000 ms/node -> ~84,000,000 ms equivalent recording
# (today's single-node record is 10,000,000 ms -- this is ~8.4x that).
# --mem=460G leaves ~40G headroom under the ~500G/node schedulable ceiling
# (measured: 400G@10M ms, extrapolated 650G@20M ms -> ~450G expected @12M ms).
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_n12500_kpool_test.slurm

#SBATCH --job-name=ca_n12500_kpool
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460G
#SBATCH --time=10:00:00
#SBATCH --array=1-7
#SBATCH --output=slurm_logs/n12500_kpool_%a_%j.log

set -eo pipefail

WS=$(ws_find calcium_ar)
PY="${GPU_PY:-$WS/env-gpu/bin/python}"
[ -x "$PY" ] || PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"

export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

NOISE_SEED=$((200 + SLURM_ARRAY_TASK_ID))
ADJ="$WS/results/wrapup_n12500r4_T10000k/seed1/adj_true.npy"

echo "host: $(hostname)  git: $(git rev-parse HEAD)  date: $(date)"
echo "array task: $SLURM_ARRAY_TASK_ID  noise_seed: $NOISE_SEED  adjacency: $ADJ"
"$PY" - <<'PYCHK'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available(), "No CUDA torch -- set GPU_PY to a CUDA env."
PYCHK

[ -f "$ADJ" ] || { echo "FATAL: reference adjacency not found at $ADJ"; exit 1; }

"$PY" scripts/wrapup_run_stream.py --net n12500_r4 \
      --adjacency-file "$ADJ" \
      --sweep 12000000 \
      --seeds "$NOISE_SEED" \
      --methods ols --device cuda --chunk-ms 5000 \
      --cache-dir "$WS/gt_cache" --save-moments

echo "task $SLURM_ARRAY_TASK_ID done: $(date)"
