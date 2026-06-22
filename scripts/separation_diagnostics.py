"""
Three-class separation diagnostics for the connectivity estimate.

The core difficulty of this problem is that the inferred weights A[i,j] fall into
three TRUE classes — excitatory (+), unconnected (0), inhibitory (-) — whose
value distributions OVERLAP. Detection errors come from the (connected vs none)
overlap; type errors from the (E vs I) overlap. This script measures that overlap
and shows how each pipeline step and each lambda setting changes it.

Metrics
-------
- Overlap coefficient OVL(p,q) = integral min(p,q)  in [0,1]; 0 = disjoint (good),
  1 = identical (bad). Computed on shared histogram bins.
    * detection overlap : |A| of connected  vs  |A| of unconnected
    * type overlap      :  A  of true-E      vs   A  of true-I   (connected only)
- Bayes accuracy ceiling: the best 3-class accuracy ANY decision rule could get
  using only the scalar A[i,j] = integral max_c (prior_c * p(A|c)). This is the
  "highest reachable performance" for that estimate's value distribution.
- Native 3-class confusion matrix: the method's own decision (A==0 -> none,
  else sign), with per-class recall (TPR) and overall accuracy.

Outputs
-------
- printed table (per pipeline step) + confusion matrices
- results/separation_diagnostics/distributions.png  — A by true class, per step
- results/separation_diagnostics/lambda_sweep.png    — overlap/ceiling vs lambda1

Usage:  python scripts/separation_diagnostics.py
"""

import sys
from pathlib import Path

import numpy as np
import zarr
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from calcium_ar.solvers.fista import solve as fista_solve

DS = "results/solver_comparison_N100/dataset"
FEED = "results/regularization_test/feed.zarr"
OUT = Path("results/separation_diagnostics")
NE = 80
LAG = 15
LAM1, LAM2 = 3e-3, 1e-3
N_ITER = 500
NBINS = 200


# ----------------------------------------------------------------- solvers ---- #
def strongest_entry_types(A):
    N = A.shape[0]; t = np.ones(N)
    for j in range(N):
        c = A[:, j].copy(); c[j] = 0.0
        t[j] = np.sign(c[np.argmax(np.abs(c))]) or 1.0
    return t


def hard_dale_fista(Cxx, Cyx, types, lam1=LAM1, lam2=LAM2, n_iter=N_ITER):
    N = Cxx.shape[0]
    L = float(np.linalg.eigvalsh(Cxx)[-1]) + lam2
    step = 1.0 / L
    A = np.zeros((N, N)); Z = A.copy(); tk = 1.0
    tcol = types[None, :]
    for _ in range(n_iter):
        grad = Z @ Cxx - Cyx + lam2 * Z
        u = Z - step * grad
        V = np.sign(u) * np.maximum(np.abs(u) - step * lam1, 0.0)
        V = tcol * np.maximum(tcol * V, 0.0)
        tnew = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
        Z = V + ((tk - 1.0) / tnew) * (V - A)
        A, tk = V, tnew
    np.fill_diagonal(A, 0.0)
    return A


# ----------------------------------------------------------- overlap / Bayes ---- #
def _grid(values):
    lo, hi = np.percentile(values, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1e-9
    return np.linspace(lo, hi, NBINS + 1)


def overlap_coefficient(a, b):
    """Integral of min(density_a, density_b) on shared bins. 0=disjoint, 1=identical."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    edges = _grid(np.concatenate([a, b]))
    bw = edges[1] - edges[0]
    pa, _ = np.histogram(a, bins=edges, density=True)
    pb, _ = np.histogram(b, bins=edges, density=True)
    return float(np.minimum(pa, pb).sum() * bw)


def ceiling_from_overlap(ovl):
    """Best balanced 2-class accuracy of ANY single-threshold rule = 1 - OVL/2 (exact)."""
    return 1.0 - 0.5 * ovl


def neuron_type_accuracy(A, NE):
    """Per-NEURON (column) type via strongest-entry rule vs truth. Returns (overall, inh-recall)."""
    pred = strongest_entry_types(A)
    true = np.where(np.arange(A.shape[0]) < NE, 1.0, -1.0)
    overall = float(np.mean(pred == true))
    inh = true == -1
    inh_rec = float(np.mean(pred[inh] == true[inh])) if inh.any() else float("nan")
    return overall, inh_rec


def confusion3(A, true_class, off):
    """Native per-EDGE 3-class confusion: pred = sign(A) (0 if exactly zero), off-diagonal only."""
    pred = np.sign(A).astype(int)[off]
    tc = true_class[off]
    order = [1, 0, -1]                      # E, none, I
    M = np.zeros((3, 3), dtype=int)
    for ti, t in enumerate(order):
        for pi, p in enumerate(order):
            M[ti, pi] = int(np.sum((tc == t) & (pred == p)))
    recall = np.array([M[i, i] / M[i].sum() if M[i].sum() else np.nan for i in range(3)])
    return M, recall, order


# --------------------------------------------------------------------- main ---- #
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Convention: inferred A[i,j] = effect j->i, aligned to adj_true.T (matches metrics.py).
    adj = np.load(Path(DS) / "adj_true.npy").T.copy(); np.fill_diagonal(adj, 0.0)
    feed = np.asarray(zarr.open(FEED, "r")[:])
    N = adj.shape[0]; off = ~np.eye(N, dtype=bool)

    # true class per off-diagonal entry: +1 exc, 0 none, -1 inh
    true_class = np.sign(adj).astype(int)
    tc = true_class[off]

    X = feed[:, :-LAG] - feed[:, :-LAG].mean(1, keepdims=True)
    Y = feed[:, LAG:] - feed[:, LAG:].mean(1, keepdims=True)
    M = X.shape[1]
    Cxx = (X @ X.T) / M
    Cyx = (Y @ X.T) / M

    def metrics_row(A):
        a = A[off]; av = np.abs(a)
        det_ovl = overlap_coefficient(av[tc != 0], av[tc == 0])   # connected vs none (|A|)
        det_auc = roc_auc_score((tc != 0).astype(int), av)        # ranking separability
        edge = tc != 0                                            # restrict type to true edges
        type_ovl = overlap_coefficient(a[tc == 1], a[tc == -1])   # E vs I (signed)
        type_auc = roc_auc_score((tc[edge] == 1).astype(int), a[edge])
        ntype, ninh = neuron_type_accuracy(A, NE)                 # per-NEURON (column) type
        return dict(det_ovl=det_ovl, det_auc=det_auc, type_ovl=type_ovl,
                    type_auc=type_auc, ntype=ntype, ninh=ninh)

    # ---- pipeline steps at the frozen lambda ----
    A_ols = Cyx @ np.linalg.inv(Cxx + 1e-9 * np.eye(N)); np.fill_diagonal(A_ols, 0.0)
    A_en = fista_solve(FEED, lag=LAG, lam_l1=LAM1, lam_l2=LAM2, n_iter=N_ITER, chunk_size=10000)
    np.fill_diagonal(A_en, 0.0)
    A_dale = hard_dale_fista(Cxx, Cyx, strongest_entry_types(A_en))

    steps = [("OLS", A_ols), ("EN(3e-3)", A_en), ("EN+Dale (C1)", A_dale)]

    print("DETECTION = connected-vs-none (|A|) ; TYPE = E-vs-I (signed).")
    print("OVL: overlap 0..1 (lower better). thr-acc = 1-OVL/2 = best single-threshold "
          "balanced acc. AUC = ranking separability. neuron-type = per-COLUMN type.\n")
    print(f"{'step':>16} | {'det_OVL':>8}{'det_thr':>8}{'det_AUC':>8} | "
          f"{'typ_OVL':>8}{'typ_thr':>8}{'typ_AUC':>8} | {'NEUR_typ':>9}{'inh_rec':>8}")
    print("-" * 92)
    for name, A in steps:
        r = metrics_row(A)
        print(f"{name:>16} | {r['det_ovl']:>8.3f}{ceiling_from_overlap(r['det_ovl']):>8.3f}"
              f"{r['det_auc']:>8.3f} | {r['type_ovl']:>8.3f}{ceiling_from_overlap(r['type_ovl']):>8.3f}"
              f"{r['type_auc']:>8.3f} | {r['ntype']:>9.3f}{r['ninh']:>8.3f}")

    print("\nNative per-EDGE 3-class confusion (rows=TRUE E/none/I, cols=PRED E/none/I)."
          "\n  Note: low per-edge I-recall but high per-NEURON type = the region-vs-edge duality.")
    for name, A in steps:
        Mc, recall, _ = confusion3(A, true_class, off)
        print(f"\n  {name}  (per-edge recall E/0/I = {recall[0]:.2f}/{recall[1]:.2f}/{recall[2]:.2f}):")
        for ti, lab in enumerate(["E ", "0 ", "I "]):
            print(f"    true {lab} " + " ".join(f"{Mc[ti, pi]:7d}" for pi in range(3)))

    # ---- distributions plot ----
    fig, axes = plt.subplots(1, len(steps), figsize=(5 * len(steps), 4), sharey=True)
    colors = {1: "tab:red", 0: "0.6", -1: "tab:blue"}
    names_c = {1: "E (+)", 0: "none (0)", -1: "I (-)"}
    for ax, (name, A) in zip(axes, steps):
        a = A[off]
        rng = np.percentile(a, [1, 99])
        bins = np.linspace(rng[0], rng[1], 80)
        for c in (1, 0, -1):
            v = a[tc == c]
            if len(v):
                ax.hist(v, bins=bins, density=True, histtype="step", lw=1.8,
                        color=colors[c], label=names_c[c])
        ax.set_title(name); ax.set_xlabel("inferred weight  A[i,j]"); ax.set_yscale("log")
    axes[0].set_ylabel("density (log)"); axes[0].legend()
    fig.suptitle("Inferred-weight distribution by TRUE class — overlap = the problem")
    fig.tight_layout(); fig.savefig(OUT / "distributions.png", dpi=130); plt.close(fig)

    # ---- lambda sweep (EN+Dale) ----
    lams = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    det_ovl, det_auc, type_auc, ntype = [], [], [], []
    for lam in lams:
        A = fista_solve(FEED, lag=LAG, lam_l1=lam, lam_l2=LAM2, n_iter=N_ITER, chunk_size=10000)
        np.fill_diagonal(A, 0.0)
        Ad = hard_dale_fista(Cxx, Cyx, strongest_entry_types(A), lam1=lam)
        a = Ad[off]; av = np.abs(a); edge = tc != 0
        det_ovl.append(overlap_coefficient(av[tc != 0], av[tc == 0]))
        det_auc.append(roc_auc_score((tc != 0).astype(int), av))
        type_auc.append(roc_auc_score((tc[edge] == 1).astype(int), a[edge]))
        ntype.append(neuron_type_accuracy(Ad, NE)[0])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(lams, det_ovl, "o-", label="detection overlap (lower=better)")
    ax.plot(lams, det_auc, "^-", label="detection AUC (higher=better)")
    ax.plot(lams, type_auc, "s-", label="type AUC E-vs-I (higher=better)")
    ax.plot(lams, ntype, "d-", label="per-neuron type accuracy (higher=better)")
    ax.set_xscale("log"); ax.set_xlabel("L1 strength  lambda1"); ax.set_ylabel("metric")
    ax.axvline(LAM1, color="k", ls=":", lw=1, label=f"frozen lambda1={LAM1:g}")
    ax.set_title("How L1 strength moves the three-class separation (EN+Dale)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "lambda_sweep.png", dpi=130); plt.close(fig)

    print(f"\nSaved plots to {OUT}/distributions.png and {OUT}/lambda_sweep.png")


if __name__ == "__main__":
    main()
