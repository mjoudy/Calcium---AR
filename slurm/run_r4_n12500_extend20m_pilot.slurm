#!/bin/bash
# R.4 scaling ladder, N=12500: PILOT for the 20M-ms extension.
#
# N=12500 is by far the most expensive size, and 20M ms has NEVER been run at
# this size before (previous ceiling was 10M ms, 3 seeds) -- both the time
# and memory numbers below are extrapolated, not measured. Run THIS single
# seed first, confirm it finishes within budget and doesn't OOM, THEN submit
# run_r4_n12500_extend20m_array.slurm for seeds 2-5. Do not submit the array
# job until this one has actually succeeded.
#
# SIMULATION ONLY -- deliberately does NOT call analyze_run.py, because that
# aggregates across every seed*/ dir present, and running it here (before
# seeds 2-5 exist) then again per array task would race on metrics.csv.
# Run slurm/run_r4_n12500_extend20m_analyze.slurm once ALL 5 seeds are done.
#
# COST (extrapolated): 5M ms ~= 2.5-3h/seed, 10M ms ~= 5-6h/seed (both
# measured). Roughly linear in T -> 20M ms ~= 10-12h for this one seed.
# Memory: 5M ms peaked at 240G/seed, 10M ms at 400G/seed (sequential
# processing, so it's a per-seed peak, not additive across seeds) -- 20M ms
# extrapolated to ~650G. THIS NUMBER IS THE BIGGEST UNKNOWN in the whole
# plan -- if your partition has no node with this much memory, this job will
# sit pending or fail; check `sinfo -p l40s -o "%N %m"` before submitting and
# scale --mem down / ask if a bigger-memory partition exists.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r4_n12500_extend20m_pilot.slurm

#SBATCH --job-name=ca_r4_n12500x20m_s1
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=650G
#SBATCH --time=20:00:00
#SBATCH --output=slurm_logs/r4_n12500x20m_s1_%j.log

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
PYCHK

"$PY" scripts/wrapup_run_stream.py --net n12500_r4 \
      --sweep 100000 200000 500000 1000000 2000000 5000000 7500000 10000000 \
               15000000 20000000 \
      --seeds 1 \
      --methods ols --device cuda --chunk-ms 5000 \
      --cache-dir "$WS/gt_cache" --save-moments

echo "pilot seed done: $(date)"
echo "--- check this before submitting the array job: ---"
echo "  sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State"
