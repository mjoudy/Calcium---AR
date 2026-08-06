#!/bin/bash
# R.4 scaling ladder, N=2500: EXTEND to the full 6-point grid
# (100k/200k/500k/1M/2M/5M ms) so this line spans the same x-axis range as
# N=12500 in fig_R4 (professor's feedback: lines should stretch across the
# full horizontal axis, not sit in disjoint segments).
#
# Old max was 1M ms; this needs 5M ms, a LONGER simulation than what's cached
# -> full fresh NEST resimulation (not free). N=2500 is 5x smaller than
# N=12500 (which took ~12h at 5M ms), so this should fit comfortably inside
# the time budget below.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r4_n2500_extend.slurm

#SBATCH --job-name=ca_r4_n2500x
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/r4_n2500x_%j.log

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
PYCHK

"$PY" scripts/wrapup_run_stream.py --net n2500_r4 \
      --sweep 100000 200000 500000 1000000 2000000 5000000 --seeds 1 \
      --methods ols --device cuda --chunk-ms 5000 \
      --cache-dir "$WS/gt_cache" --save-moments

for Tk in 100 200 500 1000 2000 5000; do
    "$PY" scripts/analyze_run.py --data "$WS/results/wrapup_n2500r4_T${Tk}k" \
        || echo "analyze T${Tk}k failed (continuing)"
done

echo "done: $(date)"
echo "--- copy home just these: ---"
ls -lh "$WS"/results/wrapup_n2500r4_T*k/metrics.csv 2>/dev/null
