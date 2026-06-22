"""
Method-comparison dashboard — three views across the pipeline solutions.

Selected progression (detection -> type -> magnitude):
    OLS  ->  EN(3e-3)  ->  EN+Dale (C1)  ->  EN+Dale+balance (full pipeline)

Figures (saved to results/method_comparison/):
  1. metrics.png       — every method's metrics side by side (bars) + E/I ratio vs true g.
  2. distributions.png — the three TRUE-class weight distributions, one panel per method,
                         so you can see the overlap change along the pipeline.
  3. neuron_entries.png— a random set of neurons (some E, some I); each neuron's outgoing
                         entries shown per method, to see how well a column's weight
                         distribution matches the neuron's true type.

Usage:  python scripts/method_comparison_viz.py
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
OUT = Path("results/method_comparison")
NE = 80
LAG = 15
LAM1, LAM2 = 3e-3, 1e-3
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


# --------------------------------------------------------------------- main --- #
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    adj = np.load(Path(DS) / "adj_true.npy"); np.fill_diagonal(adj, 0.0)
    adj_al = adj.T.copy(); np.fill_diagonal(adj_al, 0.0)        # aligned to inferred A[i,j]
    feed = np.asarray(zarr.open(FEED, "r")[:])
    meta = json.loads((Path(DS) / "metadata.json").read_text())["sim_params"]
    spk = np.asarray(zarr.open(str(Path(DS) / "spikes.zarr"), "r")[:])
    rates = spk.sum(1) / (meta["sim_time"] / 1000.0)
    N = adj.shape[0]; off = ~np.eye(N, dtype=bool)
    m = connectivity_metrics._metrics

    Xc = feed[:, :-LAG] - feed[:, :-LAG].mean(1, keepdims=True)
    Yc = feed[:, LAG:] - feed[:, LAG:].mean(1, keepdims=True)

    # ---- build the selected methods ----
    A_ols = (Yc @ Xc.T) @ np.linalg.inv((Xc @ Xc.T) + 1e-9 * np.eye(N)); np.fill_diagonal(A_ols, 0.0)
    A_en = fista_solve(FEED, lag=LAG, lam_l1=LAM1, lam_l2=LAM2, n_iter=500, chunk_size=10000)
    np.fill_diagonal(A_en, 0.0)
    A_dale = dale_fista(Xc, Yc, strongest_entry_types(A_en))
    A_full = rescale_balance_nz(A_dale, rates)
    methods = [("OLS", A_ols), ("EN(3e-3)", A_en), ("EN+Dale", A_dale), ("EN+Dale+balance", A_full)]

    true_class = np.sign(adj_al)                               # +1 E, 0 none, -1 I (aligned)
    tc = true_class[off]

    # ============================ FIGURE 1: metrics ============================ #
    bar_metrics = ["spearman", "pearson", "auc", "f1", "precision", "type_acc", "dale"]
    keyfn = {"spearman": "spearman", "pearson": "pearson", "auc": "auc_roc", "f1": "f1",
             "precision": "precision", "type_acc": "dale_type_accuracy", "dale": "degree_of_daleianity"}
    scores = {name: {mk: m[keyfn[mk]](A, adj) for mk in bar_metrics} for name, A in methods}

    def ei(A):
        g = adj_al[off]; a = np.abs(A[off]); inh, exc = a[g < 0], a[g > 0]
        return float(np.median(inh) / np.median(exc)) if len(inh) and len(exc) and np.median(exc) else np.nan
    ei_vals = {name: ei(A) for name, A in methods}

    fig, (axb, axe) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [3, 1]})
    x = np.arange(len(bar_metrics)); w = 0.8 / len(methods)
    for k, (name, _) in enumerate(methods):
        vals = [scores[name][mk] for mk in bar_metrics]
        axb.bar(x + k * w, vals, w, label=name)
    axb.set_xticks(x + 0.4 - w / 2); axb.set_xticklabels(bar_metrics, rotation=20)
    axb.set_ylim(0, 1); axb.set_ylabel("score"); axb.legend(fontsize=9)
    axb.set_title("Estimated metrics by method (higher = better)")
    axb.grid(axis="y", alpha=0.3)

    names = [n for n, _ in methods]
    axe.bar(names, [ei_vals[n] for n in names], color="tab:purple")
    axe.axhline(G_TRUE, color="k", ls="--", lw=1.2, label=f"true g = {G_TRUE:g}")
    axe.set_yscale("log"); axe.set_ylabel("E/I magnitude ratio"); axe.legend(fontsize=9)
    axe.set_title("E/I ratio vs truth"); axe.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(OUT / "metrics.png", dpi=130); plt.close(fig)

    # ====================== FIGURE 2: class distributions ====================== #
    colors = {1: "tab:red", 0: "0.6", -1: "tab:blue"}; cname = {1: "E (+)", 0: "none (0)", -1: "I (-)"}
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4), sharey=True)
    for ax, (name, A) in zip(axes, methods):
        a = A[off]; rng = np.percentile(a, [1, 99]); bins = np.linspace(rng[0], rng[1], 80)
        for c in (1, 0, -1):
            v = a[tc == c]
            if len(v):
                ax.hist(v, bins=bins, density=True, histtype="step", lw=1.8,
                        color=colors[c], label=cname[c])
        ax.set_title(name); ax.set_xlabel("inferred weight  A[i,j]"); ax.set_yscale("log")
    axes[0].set_ylabel("density (log)"); axes[0].legend()
    fig.suptitle("Three-class weight distribution along the pipeline (overlap = the difficulty)")
    fig.tight_layout(); fig.savefig(OUT / "distributions.png", dpi=130); plt.close(fig)

    # ==================== FIGURE 3: random neurons vs type ===================== #
    rng = np.random.default_rng(0)
    sel = list(rng.choice(NE, 3, replace=False)) + list(rng.choice(np.arange(NE, N), 3, replace=False))
    cols3 = methods[:3]                                       # sign structure (skip rescale)
    fig, axes = plt.subplots(len(sel), len(cols3), figsize=(4 * len(cols3), 2.3 * len(sel)),
                             sharex="col")
    for r, j in enumerate(sel):
        true_type = "E" if j < NE else "I"
        is_syn = adj_al[:, j] != 0; is_syn[j] = False          # true outgoing synapses of neuron j
        for cc, (mname, A) in enumerate(cols3):
            ax = axes[r, cc]
            col = A[:, j].copy(); col[j] = np.nan
            yj = lambda mask: np.full(mask.sum(), 0.0) + rng.uniform(-0.15, 0.15, mask.sum())
            non = ~is_syn & ~np.isnan(col)
            ax.scatter(col[non], 0.0 + rng.uniform(-0.15, 0.15, non.sum()), s=8, color="0.75",
                       label="non-target" if (r == 0 and cc == 0) else None)
            ax.scatter(col[is_syn], 1.0 + rng.uniform(-0.15, 0.15, is_syn.sum()), s=18,
                       color="tab:red" if true_type == "E" else "tab:blue",
                       label="true synapse" if (r == 0 and cc == 0) else None)
            ax.axvline(0, color="k", lw=0.8)
            ax.set_yticks([0, 1]); ax.set_yticklabels(["non", "syn"], fontsize=7)
            if r == 0:
                ax.set_title(mname)
            if cc == 0:
                ax.set_ylabel(f"neuron {j}\n(TRUE {true_type})", fontsize=9)
    for cc in range(len(cols3)):
        axes[-1, cc].set_xlabel("inferred outgoing weight")
    fig.legend(loc="upper right", fontsize=9)
    fig.suptitle("Random neurons: do their outgoing-weight entries match their type?")
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(OUT / "neuron_entries.png", dpi=130); plt.close(fig)

    print("methods:", ", ".join(n for n, _ in methods))
    print("E/I ratios:", {n: round(ei_vals[n], 3) for n in names}, " (true g =", G_TRUE, ")")
    print(f"Saved: {OUT}/metrics.png, distributions.png, neuron_entries.png")


if __name__ == "__main__":
    main()
