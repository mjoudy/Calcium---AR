"""
Full per-method dashboards for every pipeline solution.

For EACH method (OLS -> EN -> EN+Dale -> EN+Dale+balance) two figures are saved
to results/method_dashboards/:

  dashboard_<method>.png  — 6 panels:
      top:    ground-truth G (adj_true.T) | estimated A | |A - G_scaled|
      bottom: 3-class confusion matrix | weight distributions by true class | true-vs-est scatter
  neurons_<method>.png    — column distributions of A: a few E and I neurons, each a
      bar plot of outgoing weights (red +, blue -), true connections shaded, with the
      column daleianity D(x) and +/- counts.

Plus metrics_comparison.png — all methods' metrics side by side.

Usage:  python scripts/method_dashboards.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from calcium_ar.solvers.fista import solve as fista_solve
from calcium_ar.experiments.metrics import connectivity_metrics

DS = "results/solver_comparison_N100/dataset"
FEED = "results/regularization_test/feed.zarr"
OUT = Path("results/method_dashboards")
NE = 80
LAG = 15
LAM1, LAM2 = 3e-3, 1e-3
THR_FRAC = 0.01
G_TRUE = 5.0


# ------------------------------------------------------------- reused solvers --- #
def strongest_entry_types(A):
    N = A.shape[0]; t = np.ones(N)
    for j in range(N):
        c = A[:, j].copy(); c[j] = 0.0
        t[j] = np.sign(c[np.argmax(np.abs(c))]) or 1.0
    return t


def dale_fista(X, Y, types, lam1=LAM1, lam2=LAM2, n_iter=500):
    N, M = X.shape
    Cxx = (X @ X.T) / M; Cyx = (Y @ X.T) / M
    Lip = np.linalg.eigvalsh(Cxx)[-1] + lam2; thr = lam1 / Lip
    tcol = types[None, :]
    A = np.zeros((N, N)); Z = A.copy(); tk = 1.0
    for _ in range(n_iter):
        V = Z - (Z @ Cxx - Cyx + lam2 * Z) / Lip
        V = np.sign(V) * np.maximum(np.abs(V) - thr, 0.0)
        V = tcol * np.maximum(tcol * V, 0.0)
        tnew = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        Z = V + ((tk - 1) / tnew) * (V - A); A, tk = V, tnew
    np.fill_diagonal(A, 0.0)
    return A


def balance_g(types, rates):
    E = types > 0; I = types < 0
    if E.sum() == 0 or I.sum() == 0 or rates[I].mean() == 0:
        return np.nan
    return (E.sum() * rates[E].mean()) / (I.sum() * rates[I].mean())


def rescale_balance_nz(A, rates):
    N = A.shape[0]; off = ~np.eye(N, dtype=bool); B = A.copy()
    t = strongest_entry_types(A); g = balance_g(t, rates)
    Ec, Ic = np.where(t > 0)[0], np.where(t < 0)[0]

    def med_nz(cols):
        mk = np.isin(np.arange(N), cols)[None, :] & off
        v = np.abs(A[mk]); v = v[v > 1e-9]
        return np.median(v) if v.size else 0.0

    medE, medI = med_nz(Ec), med_nz(Ic)
    if medI > 1e-9 and medE > 0 and np.isfinite(g):
        B[:, Ic] = A[:, Ic] * (g * medE / medI)
    np.fill_diagonal(B, 0.0)
    return B


# ------------------------------------------------------------------- figures --- #
def global_scale(A, G, off):
    """Least-squares scalar s minimising ||A - s G|| over off-diagonal."""
    g = G[off]; a = A[off]; denom = (g * g).sum()
    return float((a * g).sum() / denom) if denom > 0 else 0.0


def confusion3(A, G, off):
    thr = THR_FRAC * np.abs(A[off]).max() if np.abs(A[off]).max() > 0 else 0.0
    pred = np.where(np.abs(A) > thr, np.sign(A), 0).astype(int)[off]
    true = np.sign(G).astype(int)[off]
    order = [-1, 0, 1]                                   # Inhibitory, Unconnected, Excitatory
    Mc = np.zeros((3, 3), int)
    for ti, t in enumerate(order):
        for pi, p in enumerate(order):
            Mc[ti, pi] = int(np.sum((true == t) & (pred == p)))
    return Mc, thr


def dashboard(name, A, G, off):
    s = global_scale(A, G, off)
    Gs = s * G
    vG = np.abs(G[off]).max(); vA = np.percentile(np.abs(A[off]), 99) or 1e-3
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))

    im0 = ax[0, 0].imshow(G, cmap="RdBu_r", vmin=-vG, vmax=vG, aspect="auto")
    ax[0, 0].set_title("Ground truth  G  (adj_true.T, j→i)"); fig.colorbar(im0, ax=ax[0, 0], fraction=0.046)
    im1 = ax[0, 1].imshow(A, cmap="RdBu_r", vmin=-vA, vmax=vA, aspect="auto")
    ax[0, 1].set_title(f"Estimated  Â  ({name})"); fig.colorbar(im1, ax=ax[0, 1], fraction=0.046)
    D = np.abs(A - Gs)
    im2 = ax[0, 2].imshow(D, cmap="hot_r", vmin=0, vmax=np.percentile(D[off], 99), aspect="auto")
    ax[0, 2].set_title("|Â − G_scaled|  (abs diff)"); fig.colorbar(im2, ax=ax[0, 2], fraction=0.046)
    for a in ax[0]:
        a.set_xlabel("source neuron j"); a.set_ylabel("target neuron i")

    Mc, thr = confusion3(A, G, off)
    row = Mc / Mc.sum(1, keepdims=True).clip(1)
    im = ax[1, 0].imshow(row, cmap="Blues", vmin=0, vmax=1)
    labs = ["Inhibitory", "Unconnected", "Excitatory"]
    ax[1, 0].set_xticks(range(3)); ax[1, 0].set_xticklabels(labs, rotation=25)
    ax[1, 0].set_yticks(range(3)); ax[1, 0].set_yticklabels(labs)
    for i in range(3):
        for j in range(3):
            ax[1, 0].text(j, i, f"{row[i, j]:.2f}\n({Mc[i, j]})", ha="center", va="center",
                          color="white" if row[i, j] > 0.5 else "black", fontsize=9)
    ax[1, 0].set_title("3-class confusion (row-normalised)")
    ax[1, 0].set_xlabel("Predicted class"); ax[1, 0].set_ylabel("True class")

    a = A[off]; g = G[off]
    unc, exc, inh = a[g == 0], a[g > 0], a[g < 0]
    rng = np.percentile(a, [1, 99]); bins = np.linspace(rng[0], rng[1], 70)
    ax[1, 1].hist(unc, bins=bins, color="0.6", alpha=0.6, label=f"Unconnected (n={len(unc)})")
    ax[1, 1].hist(exc, bins=bins, color="tomato", alpha=0.65, label=f"Excitatory (n={len(exc)})")
    ax[1, 1].hist(inh, bins=bins, color="steelblue", alpha=0.65, label=f"Inhibitory (n={len(inh)})")
    ax[1, 1].axvline(thr, color="k", ls="--", lw=1); ax[1, 1].axvline(-thr, color="k", ls="--", lw=1,
                                                                      label=f"±thr={thr:.3f}")
    ax[1, 1].set_title("Inferred weight distributions by true class")
    ax[1, 1].set_xlabel("inferred weight  Â[i,j]"); ax[1, 1].set_ylabel("count"); ax[1, 1].legend(fontsize=8)

    ax[1, 2].scatter(Gs[off][g > 0], a[g > 0], s=3, alpha=0.3, color="tomato", label="Excitatory")
    ax[1, 2].scatter(Gs[off][g < 0], a[g < 0], s=3, alpha=0.3, color="steelblue", label="Inhibitory")
    ax[1, 2].scatter(Gs[off][g == 0], a[g == 0], s=2, alpha=0.15, color="0.6", label="Unconnected")
    ax[1, 2].axhline(0, color="k", lw=0.6); ax[1, 2].axvline(0, color="k", lw=0.6)
    ax[1, 2].set_xlabel("true weight  G[i,j] (rescaled)"); ax[1, 2].set_ylabel("estimated  Â[i,j]")
    ax[1, 2].set_title("True vs estimated weights"); ax[1, 2].legend(fontsize=8, markerscale=3)

    fig.suptitle(f"{name}   |   N={A.shape[0]}, lag={LAG * 0.1:g} ms", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / f"dashboard_{_slug(name)}.png", dpi=120); plt.close(fig)


def neuron_bars(name, A, G, sel):
    N = A.shape[0]
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for k, j in enumerate(sel):
        ax = axes[k // 3, k % 3]
        col = A[:, j].copy(); col[j] = 0.0
        x = np.arange(N)
        ax.bar(x[col >= 0], col[col >= 0], color="tomato", width=0.9)
        ax.bar(x[col < 0], col[col < 0], color="steelblue", width=0.9)
        for i in np.where(G[:, j] != 0)[0]:               # true outgoing synapses (ground truth)
            ax.axvspan(i - 0.5, i + 0.5, color="khaki", alpha=0.35, zorder=0)
        ax.axhline(col.mean(), color="orange", ls="--", lw=1, label=f"mean={col.mean():.3f}")
        npos, nneg = int((col > 0).sum()), int((col < 0).sum())
        D = max(npos, nneg) / (npos + nneg) if (npos + nneg) else float("nan")
        ttype = "E" if j < NE else "I"
        ax.set_title(f"Neuron {j}  [{ttype}]\nD(x)={D:.3f}   +:{npos}  −:{nneg}", fontsize=10)
        ax.set_xlabel("target neuron i"); ax.set_ylabel("Â[i,j]"); ax.legend(fontsize=8)
    fig.suptitle(f"Column distributions of Â — outgoing weights per neuron   ({name})\n"
                 f"(bar = target i; red +, blue −; shaded = true connection)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / f"neurons_{_slug(name)}.png", dpi=120); plt.close(fig)


def _slug(s):
    return s.replace("(", "").replace(")", "").replace("+", "_").replace(" ", "").replace("-", "")


# --------------------------------------------------------------------- main --- #
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    adj = np.load(Path(DS) / "adj_true.npy"); np.fill_diagonal(adj, 0.0)
    G = adj.T.copy(); np.fill_diagonal(G, 0.0)                 # aligned to inferred A[i,j]
    feed = np.asarray(zarr.open(FEED, "r")[:])
    meta = json.loads((Path(DS) / "metadata.json").read_text())["sim_params"]
    spk = np.asarray(zarr.open(str(Path(DS) / "spikes.zarr"), "r")[:])
    rates = spk.sum(1) / (meta["sim_time"] / 1000.0)
    N = adj.shape[0]; off = ~np.eye(N, dtype=bool)
    m = connectivity_metrics._metrics

    Xc = feed[:, :-LAG] - feed[:, :-LAG].mean(1, keepdims=True)
    Yc = feed[:, LAG:] - feed[:, LAG:].mean(1, keepdims=True)

    A_ols = (Yc @ Xc.T) @ np.linalg.inv((Xc @ Xc.T) + 1e-9 * np.eye(N)); np.fill_diagonal(A_ols, 0.0)
    A_en = fista_solve(FEED, lag=LAG, lam_l1=LAM1, lam_l2=LAM2, n_iter=500, chunk_size=10000)
    np.fill_diagonal(A_en, 0.0)
    A_dale = dale_fista(Xc, Yc, strongest_entry_types(A_en))
    A_full = rescale_balance_nz(A_dale, rates)
    methods = [("OLS", A_ols), ("EN(3e-3)", A_en), ("EN+Dale", A_dale), ("EN+Dale+balance", A_full)]

    rng = np.random.default_rng(0)
    sel = list(rng.choice(NE, 3, replace=False)) + list(rng.choice(np.arange(NE, N), 3, replace=False))

    # per-method dashboards + neuron bars
    for name, A in methods:
        dashboard(name, A, G, off)
        neuron_bars(name, A, G, sel)

    # metrics comparison
    bar_metrics = ["spearman", "pearson", "auc", "f1", "precision", "type_acc", "dale"]
    keyfn = {"spearman": "spearman", "pearson": "pearson", "auc": "auc_roc", "f1": "f1",
             "precision": "precision", "type_acc": "dale_type_accuracy", "dale": "degree_of_daleianity"}

    def ei(A):
        g = G[off]; a = np.abs(A[off]); ih, ex = a[g < 0], a[g > 0]
        return float(np.median(ih) / np.median(ex)) if len(ih) and len(ex) and np.median(ex) else np.nan

    fig, (axb, axe) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [3, 1]})
    x = np.arange(len(bar_metrics)); w = 0.8 / len(methods)
    for k, (name, A) in enumerate(methods):
        axb.bar(x + k * w, [m[keyfn[mk]](A, adj) for mk in bar_metrics], w, label=name)
    axb.set_xticks(x + 0.4 - w / 2); axb.set_xticklabels(bar_metrics, rotation=20)
    axb.set_ylim(0, 1); axb.set_ylabel("score"); axb.legend(fontsize=9); axb.grid(axis="y", alpha=0.3)
    axb.set_title("Estimated metrics by method")
    names = [n for n, _ in methods]
    axe.bar(names, [ei(A) for _, A in methods], color="tab:purple")
    axe.axhline(G_TRUE, color="k", ls="--", lw=1.2, label=f"true g={G_TRUE:g}")
    axe.set_yscale("log"); axe.set_ylabel("E/I ratio"); axe.legend(fontsize=9)
    axe.set_title("E/I ratio vs truth"); axe.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(OUT / "metrics_comparison.png", dpi=130); plt.close(fig)

    print("methods:", ", ".join(names))
    print(f"Saved per-method dashboards + neuron figures and metrics_comparison.png to {OUT}/")


if __name__ == "__main__":
    main()
