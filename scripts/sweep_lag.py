"""
Sweep over lag_ms to find the effect of AR lag on connectivity inference.

Generates the dataset once (saved to results/lag_sweep/dataset/) and runs
the ridge solver for each lag value, reusing the same ground truth data.

Usage
-----
    python scripts/sweep_lag.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt

from calcium_ar.experiments.config import ExperimentConfig
from calcium_ar.experiments.sweep import ParameterSweep

# ------------------------------------------------------------------ #
# Base configuration
# ------------------------------------------------------------------ #
base = ExperimentConfig(
    # Network — N=100, fast iteration
    n_excitatory = 80,
    n_inhibitory = 20,
    J_ex         = 10.0,        # auto-scaled for N=100: J * C_E = 80
    sim_time     = 50_000.0,    # ms
    dt           = 0.1,         # ms

    # Calcium
    tau          = 100.0,       # ms
    sigma_extra  = 0.05,

    # Preprocessing
    smooth_window_ms = 3.1,
    tau_method       = "ransac",

    # Solver — keep ridge fixed while varying lag
    solver       = "ridge",
    lag_ms       = 10.0,        # overridden by sweep
    lam          = 1.0,
    chunk_size   = 10_000,

    # Reuse the same dataset across all lag values
    data_path    = "results/lag_sweep/dataset",

    name         = "lag_sweep",
    seed         = 42,
    output_dir   = "results/lag_sweep",
)

# ------------------------------------------------------------------ #
# Sweep
# ------------------------------------------------------------------ #
LAG_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

sweep = ParameterSweep(
    params     = {"lag_ms": LAG_VALUES},
    base_config = base,
    sweep_dir   = "results/lag_sweep",
)

print(f"Sweeping lag_ms over {LAG_VALUES} ms")
print(f"Each lag_ms → lag_samples = round(lag_ms / {base.dt}) at dt={base.dt}ms")
for v in LAG_VALUES:
    print(f"  lag_ms={v:5.1f}  →  {max(1, round(v / base.dt)):4d} samples")
print()

# Generate the dataset once before launching parallel workers so they don't
# race each other trying to write to the same zarr directory.
print("=== Step 1: ensure dataset exists ===")
from calcium_ar.data.dataset import SimulatedDataset
SimulatedDataset.load_or_generate(base, base.data_path)
print()

print("=== Step 2: run sweep (solver only, dataset reused) ===")
df = sweep.run(n_jobs=-1)

# ------------------------------------------------------------------ #
# Display results
# ------------------------------------------------------------------ #
keep = ["lag_ms", "connectivity/pearson", "connectivity/auc_roc",
        "connectivity/mse", "tau/tau_mae"]
available = [c for c in keep if c in df.columns]
print("\n=== Results ===")
print(df[available].sort_values("lag_ms").to_string(index=False))

# ------------------------------------------------------------------ #
# Plot
# ------------------------------------------------------------------ #
df_sorted = df.sort_values("lag_ms")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(df_sorted["lag_ms"], df_sorted["connectivity/pearson"],
             marker="o", linewidth=1.5)
axes[0].set_xlabel("lag (ms)")
axes[0].set_ylabel("Pearson correlation")
axes[0].set_title("Connectivity inference vs lag")
axes[0].grid(True, alpha=0.4)

axes[1].plot(df_sorted["lag_ms"], df_sorted["connectivity/auc_roc"],
             marker="o", color="C1", linewidth=1.5)
axes[1].set_xlabel("lag (ms)")
axes[1].set_ylabel("AUC-ROC")
axes[1].set_title("AUC-ROC vs lag")
axes[1].grid(True, alpha=0.4)

plt.tight_layout()
out_path = "results/lag_sweep/lag_sweep_results.png"
os.makedirs("results/lag_sweep", exist_ok=True)
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved → {out_path}")
plt.show()
