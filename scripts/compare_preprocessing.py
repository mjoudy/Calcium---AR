"""
Compare OLS on raw calcium vs OLS on preprocessed feed.

Raw calcium:   the solver reads the calcium zarr directly — no smoothing,
               no tau estimation, no feed reconstruction.
Preprocessed:  normal pipeline (SG smooth → RANSAC tau → feed = deriv + C/tau).

Both use chunked OLS (numpy).  No new simulation — reuses lag_sweep/dataset.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import zarr
import matplotlib.pyplot as plt

from calcium_ar.solvers.chunked_ols import solve as ols_solve
from calcium_ar.experiments.config import ExperimentConfig
from calcium_ar.experiments.runner import run_single
from calcium_ar.experiments.metrics import compute_all as compute_metrics

DATASET = "results/lag_sweep/dataset"
OUTPUT  = "results/preprocessing_comparison"
LAG_MS  = 2.0
DT      = 0.1

os.makedirs(OUTPUT, exist_ok=True)

adj_true = np.load(os.path.join(DATASET, "adj_true.npy"))
np.fill_diagonal(adj_true, 0.0)

lag_samples = max(1, round(LAG_MS / DT))

# ------------------------------------------------------------------ #
# Case 1: raw calcium — no preprocessing at all
# ------------------------------------------------------------------ #
print("=== OLS on raw calcium (no preprocessing) ===")
t0 = time.perf_counter()
A_raw = ols_solve(
    os.path.join(DATASET, "calcium.zarr"),
    lag=lag_samples,
    chunk_size=10_000,
)
t_raw = time.perf_counter() - t0
np.fill_diagonal(A_raw, 0.0)
m_raw = compute_metrics(A_raw, adj_true)
print(f"done in {t_raw:.1f}s  "
      f"pearson={m_raw['connectivity/pearson']:.3f}  "
      f"auc={m_raw['connectivity/auc_roc']:.3f}")

# ------------------------------------------------------------------ #
# Case 2: preprocessed feed — normal pipeline
# ------------------------------------------------------------------ #
print("\n=== OLS on preprocessed feed ===")
cfg = ExperimentConfig(
    n_excitatory     = 80,
    n_inhibitory     = 20,
    J_ex             = 10.0,
    sim_time         = 50_000.0,
    dt               = DT,
    tau              = 100.0,
    sigma_extra      = 0.05,
    smooth_window_ms = 3.1,
    tau_method       = "ransac",
    lag_ms           = LAG_MS,
    chunk_size       = 10_000,
    data_path        = DATASET,
    output_dir       = OUTPUT,
    seed             = 42,
    solver           = "ols",
    name             = "ols_feed",
)
t0 = time.perf_counter()
r_feed = run_single(cfg)
t_feed = time.perf_counter() - t0
A_feed = np.load(os.path.join(r_feed.run_dir, "adj_inferred.npy"))
m_feed = r_feed.metrics

# ------------------------------------------------------------------ #
# Print summary table
# ------------------------------------------------------------------ #
print("\n" + "=" * 58)
print(f"{'':30s}  {'raw calcium':>12}  {'feed (prep)':>12}")
print("-" * 58)
for key, label in [
    ("connectivity/pearson",  "Pearson"),
    ("connectivity/spearman", "Spearman"),
    ("connectivity/auc_roc",  "AUC-ROC"),
    ("connectivity/mse",      "MSE"),
]:
    print(f"  {label:28s}  {m_raw[key]:12.4f}  {m_feed[key]:12.4f}")
print("-" * 58)
print(f"  {'Runtime (s)':28s}  {t_raw:12.1f}  {t_feed:12.1f}")
print("=" * 58)

# ------------------------------------------------------------------ #
# Plot
# ------------------------------------------------------------------ #
N = adj_true.shape[0]
mask = ~np.eye(N, dtype=bool)

gt_flat  = adj_true.T[mask]      # runner uses adj_true.T convention
A_r_flat = A_raw[mask]
A_f_flat = A_feed[mask]

# Three-class masks on the ground truth
exc_mask = gt_flat  > 0
inh_mask = gt_flat  < 0
unc_mask = gt_flat == 0

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# ---- Row 0: entry distributions per ground-truth class ----
bins = 80
for col, (A_flat, title) in enumerate([
    (A_r_flat, "Raw calcium"),
    (A_f_flat, "Preprocessed feed"),
]):
    ax = axes[0, col]
    ax.hist(A_flat[unc_mask], bins=bins, alpha=0.6, label="Unconnected",  density=True)
    ax.hist(A_flat[exc_mask], bins=bins, alpha=0.6, label="Excitatory",   density=True)
    ax.hist(A_flat[inh_mask], bins=bins, alpha=0.6, label="Inhibitory",   density=True)
    ax.set_xlabel("Estimated weight")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# ---- Row 0, col 2: metric bar chart ----
ax = axes[0, 2]
metric_labels = ["Pearson", "Spearman", "AUC-ROC"]
metric_keys_short = ["connectivity/pearson", "connectivity/spearman", "connectivity/auc_roc"]
x = np.arange(len(metric_labels))
w = 0.35
ax.bar(x - w/2, [m_raw[k]  for k in metric_keys_short], w, label="Raw",  color="C0")
ax.bar(x + w/2, [m_feed[k] for k in metric_keys_short], w, label="Feed", color="C1")
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylabel("Score")
ax.set_title("Metrics comparison")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# ---- Row 1: scatter of estimated A vs ground truth (adj_true.T) ----
for col, (A_flat, title) in enumerate([
    (A_r_flat, "Raw calcium"),
    (A_f_flat, "Preprocessed feed"),
]):
    ax = axes[1, col]
    for mask_i, label, color in [
        (unc_mask, "Unconnected", "C0"),
        (exc_mask, "Excitatory",  "C1"),
        (inh_mask, "Inhibitory",  "C2"),
    ]:
        ax.scatter(gt_flat[mask_i], A_flat[mask_i],
                   s=2, alpha=0.3, color=color, label=label, rasterized=True)
    ax.set_xlabel("True weight (adj_true.T)")
    ax.set_ylabel("Estimated weight A")
    ax.set_title(title)
    ax.legend(fontsize=8, markerscale=4)
    ax.grid(True, alpha=0.3)

# ---- Row 1, col 2: runtime bar ----
ax = axes[1, 2]
ax.bar(["Raw\ncalcium", "Preprocessed\nfeed"], [t_raw, t_feed],
       color=["C0", "C1"], width=0.5)
ax.set_ylabel("Wall time (s)")
ax.set_title("Runtime")
ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate([t_raw, t_feed]):
    ax.text(i, v + 0.3, f"{v:.1f}s", ha="center", fontsize=10)

plt.suptitle(
    f"OLS: raw calcium vs preprocessed feed  |  lag_ms={LAG_MS}  |  N=100  |  T=500k",
    fontsize=12,
)
plt.tight_layout()
out = os.path.join(OUTPUT, "preprocessing_comparison.png")
plt.savefig(out, dpi=150)
print(f"\nPlot saved → {out}")
plt.show()
