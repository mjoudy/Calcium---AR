"""
Compare chunked OLS (numpy normal equations) vs sklearn OLS (LinearRegression)
on the existing lag_sweep dataset (N=100, T=500k, no simulation needed).

Both solvers are mathematically equivalent (same intercept correction), so
the main interest is:
  - runtime difference
  - numerical agreement of the two estimated A matrices
  - connectivity metrics (should match)
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from calcium_ar.experiments.config import ExperimentConfig
from calcium_ar.experiments.runner import run_single

DATASET   = "results/lag_sweep/dataset"
OUTPUT    = "results/solver_comparison"
LAG_MS    = 2.0

os.makedirs(OUTPUT, exist_ok=True)

base = ExperimentConfig(
    n_excitatory     = 80,
    n_inhibitory     = 20,
    J_ex             = 10.0,
    sim_time         = 50_000.0,
    dt               = 0.1,
    tau              = 100.0,
    sigma_extra      = 0.05,
    smooth_window_ms = 3.1,
    tau_method       = "ransac",
    lag_ms           = LAG_MS,
    chunk_size       = 10_000,
    data_path        = DATASET,
    output_dir       = OUTPUT,
    seed             = 42,
)

# ------------------------------------------------------------------ #
# Run both solvers
# ------------------------------------------------------------------ #
print("=== Chunked OLS (numpy) ===")
t0  = time.perf_counter()
r_np = run_single(base.replace(solver="ols", name="chunked_ols"))
t_np = time.perf_counter() - t0

print(f"\n=== sklearn OLS (LinearRegression) ===")
t0   = time.perf_counter()
r_sk = run_single(base.replace(solver="sklearn_ols", name="sklearn_ols"))
t_sk = time.perf_counter() - t0

# ------------------------------------------------------------------ #
# Load the saved A matrices for direct comparison
# ------------------------------------------------------------------ #
A_np = np.load(os.path.join(r_np.run_dir, "adj_inferred.npy"))
A_sk = np.load(os.path.join(r_sk.run_dir, "adj_inferred.npy"))

max_diff  = np.abs(A_np - A_sk).max()
mean_diff = np.abs(A_np - A_sk).mean()
corr_AA   = np.corrcoef(A_np.ravel(), A_sk.ravel())[0, 1]

# ------------------------------------------------------------------ #
# Report
# ------------------------------------------------------------------ #
print("\n" + "=" * 58)
print(f"{'':30s}  {'chunked_ols':>12}  {'sklearn_ols':>12}")
print("-" * 58)

metric_keys = [
    ("connectivity/pearson",  "Pearson"),
    ("connectivity/spearman", "Spearman"),
    ("connectivity/auc_roc",  "AUC-ROC"),
    ("connectivity/mse",      "MSE"),
]
for key, label in metric_keys:
    v_np = r_np.metrics.get(key, float("nan"))
    v_sk = r_sk.metrics.get(key, float("nan"))
    print(f"  {label:28s}  {v_np:12.4f}  {v_sk:12.4f}")

print("-" * 58)
print(f"  {'Runtime (s)':28s}  {t_np:12.1f}  {t_sk:12.1f}")
print("=" * 58)
print(f"\nNumerical agreement between the two A matrices:")
print(f"  max |A_np - A_sk|  = {max_diff:.2e}")
print(f"  mean|A_np - A_sk|  = {mean_diff:.2e}")
print(f"  corr(A_np, A_sk)   = {corr_AA:.8f}")

# ------------------------------------------------------------------ #
# Plot: scatter of A entries + entry distributions
# ------------------------------------------------------------------ #
N = A_np.shape[0]
mask = ~np.eye(N, dtype=bool)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# --- scatter of off-diagonal entries ---
ax = axes[0]
ax.scatter(A_np[mask], A_sk[mask], s=2, alpha=0.3, rasterized=True)
lim = max(np.abs(A_np[mask]).max(), np.abs(A_sk[mask]).max()) * 1.05
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8)
ax.set_xlabel("A  chunked OLS")
ax.set_ylabel("A  sklearn OLS")
ax.set_title(f"Entry scatter\ncorr={corr_AA:.6f}  max_diff={max_diff:.1e}")
ax.grid(True, alpha=0.3)

# --- residuals histogram ---
ax = axes[1]
diffs = (A_np - A_sk)[mask]
ax.hist(diffs, bins=80, color="steelblue", edgecolor="none")
ax.set_xlabel("A_np - A_sk (off-diagonal)")
ax.set_title(f"Residuals\nmean={mean_diff:.1e}  max={max_diff:.1e}")
ax.grid(True, alpha=0.3)

# --- runtime bar ---
ax = axes[2]
ax.bar(["chunked\nOLS", "sklearn\nOLS"], [t_np, t_sk],
       color=["C0", "C1"], width=0.5)
ax.set_ylabel("wall time (s)")
ax.set_title("Runtime")
ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate([t_np, t_sk]):
    ax.text(i, v + 0.5, f"{v:.1f}s", ha="center", fontsize=10)

plt.suptitle(
    f"Chunked OLS vs sklearn OLS  |  lag_ms={LAG_MS}  |  N=100  |  T=500k",
    fontsize=11,
)
plt.tight_layout()
out = os.path.join(OUTPUT, "ols_comparison.png")
plt.savefig(out, dpi=150)
print(f"\nPlot saved → {out}")
plt.show()
