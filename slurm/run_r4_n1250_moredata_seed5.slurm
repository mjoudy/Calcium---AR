#!/bin/bash
# Follow-up to run_r4_n1250_moredata.slurm (job 7690782, TIMEOUT at 8h): seeds
# 1-4 completed fully, seed 5 got through checkpoint 10,000,000ms before being
# killed -- only the 15M/20M ms checkpoints are missing. This reruns seed 5
# ONLY. Its ground truth is already cached (see the 7690782 log:
# "cached ground truth -> n1250_r4_seed5_T20000k_...npz"), so NEST is skipped
# and this is just the streaming/checkpoint pass again (~1.5h, not 8h) --
# it will harmlessly re-save the checkpoints seed 5 already has too.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r4_n1250_moredata_seed5.slurm

#SBATCH --job-name=ca_r4_n1250_more_s5
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/r4_n1250_more_s5_%j.log

set -eo pipefail

WS=$(ws_find calcium_ar)
PY="${GPU_PY:-$WS/env-gpu/bin/python}"
[ -x "$PY" ] || PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"

export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "host: $(hostname)   git: $(git rev-parse HEAD)   date: $(date)   PY=$PY"

"$PY" scripts/wrapup_run_stream.py --net n1250_r4 \
      --sweep 100000 200000 500000 1000000 2000000 5000000 7500000 10000000 15000000 20000000 \
      --seeds 5 \
      --methods ols --device cuda --chunk-ms 5000 \
      --cache-dir "$WS/gt_cache" --save-moments

for Tk in 100 200 500 1000 2000 5000 7500 10000 15000 20000; do
    "$PY" scripts/analyze_run.py --data "$WS/results/wrapup_n1250r4_T${Tk}k" \
        || echo "analyze T${Tk}k failed (continuing)"
done

echo "done: $(date)"
ls -lh "$WS"/results/wrapup_n1250r4_T*k/metrics.csv 2>/dev/null
