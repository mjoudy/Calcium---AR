"""
Spectral analysis of the estimated adjacency matrix.

Experiment:
  1. Load A_hat (OLS estimate) and A_true (ground-truth synaptic weights).
  2. Show baseline distributions of A_hat off-diagonal entries per class
     (exc / inh / unconn).
  3. Spectral correction: fix negative real eigenvalues of A_hat, then
     compute G_hat = log(A_corrected) / Delta.
  4. Show distributions of G_hat entries per class.
  5. Plot eigenvalue spectrum of A_hat in the complex plane.
  6. Plot the leading eigenvector — does it separate E vs I neurons?
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm
from scipy.stats import gaussian_kde
from pathlib import Path

# ------------------------------------------------------------------ #
# Paths and parameters                                                 #
# ------------------------------------------------------------------ #
RESULT_DIR = Path(
    "results/100k_fixed/solver_comparison_N100/ols_20260529_002543_471603"
)
OUT_DIR = Path("results/spectral_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Δ = lag × dt = 10 × 0.1 ms = 1 ms
DELTA = 1.0  # ms
N_EXC = 80
N_INH = 20

# ------------------------------------------------------------------ #
# Load data                                                            #
# ------------------------------------------------------------------ #
A_hat  = np.load(RESULT_DIR / "adj_inferred.npy")
A_true = np.load(RESULT_DIR / "adj_true.npy")
N = A_hat.shape[0]

print(f"Network: N={N}  (exc={N_EXC}, inh={N_INH})")
print(f"A_true unique values: {np.unique(A_true)}")
print(f"A_hat shape: {A_hat.shape}\n")

# ------------------------------------------------------------------ #
# Class masks (off-diagonal only)                                      #
# ------------------------------------------------------------------ #
off_diag   = ~np.eye(N, dtype=bool)
exc_mask   = off_diag & (A_true == 10)
inh_mask   = off_diag & (A_true == -50)
unconn_mask = off_diag & (A_true == 0)

print("Off-diagonal entry counts:")
print(f"  exc:   {exc_mask.sum()}")
print(f"  inh:   {inh_mask.sum()}")
print(f"  unconn:{unconn_mask.sum()}\n")

# ------------------------------------------------------------------ #
# Eigenvalue analysis of A_hat                                         #
# ------------------------------------------------------------------ #
vals, vecs = np.linalg.eig(A_hat)

eps = 1e-8
pos_real_mask = (vals.real >= 0) & (np.abs(vals.imag) < eps)
neg_real_mask = (vals.real < 0)  & (np.abs(vals.imag) < eps)
complex_mask  = np.abs(vals.imag) >= eps

print("Eigenvalue breakdown:")
print(f"  positive real:  {pos_real_mask.sum()}")
print(f"  negative real:  {neg_real_mask.sum()}  ← problematic for log")
print(f"  complex pairs:  {complex_mask.sum()}")

if neg_real_mask.any():
    neg_vals = vals[neg_real_mask]
    print(f"\n  Negative real eigenvalues: {np.sort(neg_vals.real)}")

spectral_radius = np.abs(vals).max()
print(f"\n  Spectral radius of A_hat: {spectral_radius:.4f}")
print(f"  (should be < 1 for stable system; > 1 → noise artefact)\n")

# ------------------------------------------------------------------ #
# Spectral correction: flip negative real eigenvalues to |λ|          #
# ------------------------------------------------------------------ #
vals_corrected = vals.copy()
vals_corrected[neg_real_mask] = np.abs(vals[neg_real_mask])

A_corrected = (vecs @ np.diag(vals_corrected) @ np.linalg.inv(vecs)).real
G_hat = logm(A_corrected).real / DELTA

print("G_hat (after spectral correction + log):")
off_diag_G = G_hat[off_diag]
print(f"  off-diag range: [{off_diag_G.min():.4f}, {off_diag_G.max():.4f}]")
print(f"  off-diag mean:  {off_diag_G.mean():.4f}")

# Imaginary residual check
imag_residual = np.abs(logm(A_corrected).imag).max()
print(f"  max imaginary residual in logm: {imag_residual:.2e}\n")

# ------------------------------------------------------------------ #
# Separability metric: Cohen's d between exc and inh                   #
# ------------------------------------------------------------------ #
def cohens_d(a, b):
    pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return abs(a.mean() - b.mean()) / (pooled_std + 1e-12)

for label, arr in [("A_hat", A_hat), ("G_hat", G_hat)]:
    exc_vals   = arr[exc_mask]
    inh_vals   = arr[inh_mask]
    unconn_vals = arr[unconn_mask]
    d_ei = cohens_d(exc_vals, inh_vals)
    d_eu = cohens_d(exc_vals, unconn_vals)
    d_iu = cohens_d(inh_vals, unconn_vals)
    print(f"Cohen's d  [{label}]")
    print(f"  exc vs inh:   {d_ei:.3f}")
    print(f"  exc vs unconn:{d_eu:.3f}")
    print(f"  inh vs unconn:{d_iu:.3f}")
    print()

# ------------------------------------------------------------------ #
# Figure 1: Eigenvalue spectrum in the complex plane                   #
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(6, 6))

theta = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, alpha=0.5, label="unit circle")

ax.scatter(vals[pos_real_mask].real, vals[pos_real_mask].imag,
           c="steelblue", s=60, label="positive real", zorder=3)
ax.scatter(vals[complex_mask].real, vals[complex_mask].imag,
           c="forestgreen", s=30, alpha=0.6, label="complex", zorder=2)
ax.scatter(vals[neg_real_mask].real, vals[neg_real_mask].imag,
           c="crimson", s=80, marker="x", lw=2, label="negative real (problematic)", zorder=4)

ax.axhline(0, color="gray", lw=0.5)
ax.axvline(0, color="gray", lw=0.5)
ax.set_xlabel("Re(λ)")
ax.set_ylabel("Im(λ)")
ax.set_title("Eigenvalue spectrum of $\\hat{A}$")
ax.legend(fontsize=8)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(OUT_DIR / "eigenvalue_spectrum.png", dpi=150)
plt.close(fig)
print("Saved: eigenvalue_spectrum.png")

# ------------------------------------------------------------------ #
# Helper: KDE plot for three classes                                   #
# ------------------------------------------------------------------ #
def plot_class_distributions(ax, matrix, title, xlabel):
    colors = {"exc": "tab:red", "inh": "tab:blue", "unconn": "tab:gray"}
    data = {
        "exc":   matrix[exc_mask],
        "inh":   matrix[inh_mask],
        "unconn": matrix[unconn_mask],
    }
    all_vals = np.concatenate(list(data.values()))
    x = np.linspace(all_vals.min(), all_vals.max(), 500)
    for name, vals_class in data.items():
        if vals_class.std() < 1e-12:
            continue
        kde = gaussian_kde(vals_class, bw_method="scott")
        ax.plot(x, kde(x), color=colors[name], label=name, lw=2)
        ax.fill_between(x, kde(x), alpha=0.15, color=colors[name])
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

# ------------------------------------------------------------------ #
# Figure 2: Before vs After distributions                              #
# ------------------------------------------------------------------ #
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

plot_class_distributions(axes[0], A_hat,
                         "Baseline: $\\hat{A}$ off-diagonal entries",
                         "$\\hat{A}_{ij}$")
plot_class_distributions(axes[1], G_hat,
                         "After spectral correction: $\\hat{G}$ off-diagonal entries\n"
                         r"$\hat{G} = \log(\hat{A}_{\rm corr})\,/\,\Delta$",
                         "$\\hat{G}_{ij}$  (ms$^{-1}$)")

fig.suptitle("Distribution of estimated entries by synaptic class", fontsize=12)
fig.tight_layout()
fig.savefig(OUT_DIR / "class_distributions.png", dpi=150)
plt.close(fig)
print("Saved: class_distributions.png")

# ------------------------------------------------------------------ #
# Figure 3: Leading eigenvector — does it separate E vs I neurons?     #
# ------------------------------------------------------------------ #
# Sort by real part descending → dominant (slowest-decaying) mode
sort_idx = np.argsort(vals.real)[::-1]
dom_vec = vecs[:, sort_idx[0]].real   # right eigenvector of dominant eigenvalue

fig, ax = plt.subplots(figsize=(10, 3))
neuron_idx = np.arange(N)
exc_idx    = neuron_idx[:N_EXC]
inh_idx    = neuron_idx[N_EXC:]

ax.bar(exc_idx, dom_vec[exc_idx], color="tab:red",  alpha=0.7, label="excitatory")
ax.bar(inh_idx, dom_vec[inh_idx], color="tab:blue", alpha=0.7, label="inhibitory")
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("Neuron index")
ax.set_ylabel("Component")
ax.set_title(f"Leading eigenvector of $\\hat{{A}}$  "
             f"(λ = {vals[sort_idx[0]].real:.4f})")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "leading_eigenvector.png", dpi=150)
plt.close(fig)
print("Saved: leading_eigenvector.png")

# ------------------------------------------------------------------ #
# Figure 4: Mean per class — before and after                          #
# ------------------------------------------------------------------ #
labels = ["exc", "inh", "unconn"]
masks  = [exc_mask, inh_mask, unconn_mask]
colors = ["tab:red", "tab:blue", "tab:gray"]

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
for ax, matrix, title in zip(axes, [A_hat, G_hat],
                              ["$\\hat{A}$", "$\\hat{G}$"]):
    means = [matrix[m].mean() for m in masks]
    stds  = [matrix[m].std()  for m in masks]
    bars  = ax.bar(labels, means, color=colors, alpha=0.8, yerr=stds,
                   capsize=5, error_kw={"lw": 1.5})
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title(f"Mean ± std  [{title}]")
    ax.set_ylabel("entry value")

fig.suptitle("Class-conditional means before and after spectral correction + log",
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / "class_means.png", dpi=150)
plt.close(fig)
print("Saved: class_means.png")

print(f"\nAll figures saved to: {OUT_DIR}/")
