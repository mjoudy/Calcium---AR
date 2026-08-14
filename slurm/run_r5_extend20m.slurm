#!/bin/bash
# R.5 (regularization vs data) — re-solve on the extended R.4 checkpoint grid.
#
# No new simulation: fig_r5_compute.py auto-discovers every wrapup_n{1250,
# 12500}r4_T*k/seed1/{Cxx,Cyx,adj_true}.npy checkpoint under --root, so once
# R.4's 20M-ms extension has produced those files this just re-solves and
# re-scores them (existing checkpoints get overwritten with the same values,
# new ones -- 7.5M/10M/15M/20M ms -- get added). Single seed only (2026-08-14
# decision: R5 stays single-seed, no shaded bands, and no new network sizes --
# just the longer x-axis).
#
# N=1250's moments already exist out to 20M ms (see run_r4_n1250_moredata.slurm,
# already run) -- this script's N=1250 half can run RIGHT NOW, no need to wait.
# N=12500's half needs at least the pilot seed (run_r4_n12500_extend20m_pilot.slurm)
# to have finished -- R5 only reads seed1, so it does NOT need the full seeds
# 2-5 array job, just the pilot.
#
# Cheap either way (re-solving cached moments, not resimulating) -- minutes,
# not hours.
#
# Submit:  cd $(ws_find calcium_ar)/Calcium---AR
#          sbatch slurm/run_r5_extend20m.slurm

#SBATCH --job-name=ca_r5_extend20m
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/r5_extend20m_%j.log

set -eo pipefail

WS=$(ws_find calcium_ar)
PY="${GPU_PY:-$WS/env-gpu/bin/python}"
[ -x "$PY" ] || PY="$WS/env/bin/python"
cd "$WS/Calcium---AR"

export CALCIUM_AR_WORKDIR="$WS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "host: $(hostname)   git: $(git rev-parse HEAD)   date: $(date)   PY=$PY"

R5OUT="$WS/results/r5"
"$PY" scripts/fig_r5_compute.py --root "$WS/results" --net n1250r4  --device cpu  --out-dir "$R5OUT"
if ls "$WS"/results/wrapup_n12500r4_T*k/seed1/Cxx.npy >/dev/null 2>&1; then
    "$PY" scripts/fig_r5_compute.py --root "$WS/results" --net n12500r4 --device cuda --out-dir "$R5OUT"
else
    echo "[skip] no N=12500 seed1 moments found yet -- run the pilot job first"
fi

echo "done: $(date)"
echo "--- copy home just these: ---"
ls -lh "$R5OUT"/*.csv 2>/dev/null
echo
echo "rsync -avhz \${USER}@login1.nemo.uni-freiburg.de:$R5OUT/ ~/calcium_results/r5/"
