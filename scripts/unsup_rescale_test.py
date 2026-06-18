"""
Unsupervised rescale on the regularized matrix — can we drop the oracle?

The combine test showed regularize→rescale works, but the rescale used TRUE types.
Earlier the unsupervised type-guess failed on the raw OLS matrix (inhibitory neurons
misclassified, circular).  Hypothesis: regularization cleans the estimate enough that
the column-sign type-guess now works, so an UNSUPERVISED rescale matches the oracle.

We measure two things:
  (1) type-inference accuracy (overall + inhibitory-only) on raw OLS vs regularized
  (2) end metrics for OLS, EN, EN+typeeq (unsupervised), EN+oracle (ceiling)

Note on the E/I target: an unsupervised rescale can only EQUALISE the two groups
(ratio → 1); it cannot know the true biological ratio is 5 without external g.  So we
judge it on detection + rank (F1, AUC, Spearman, Pearson), not on hitting E/I = 5.

Usage:  python scripts/unsup_rescale_test.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calcium_ar.solvers.fista import solve as fista_solve
from calcium_ar.experiments.metrics import connectivity_metrics
from calcium_ar.experiments.config import ExperimentConfig
from calcium_ar.experiments.result import ExperimentResult
from calcium_ar.experiments.ledger import append_row

DS = "results/solver_comparison_N100/dataset"
FEED = "results/regularization_test/feed.zarr"
OUT = "results/unsup_rescale_test"
NE = 80
LAG = 15


def ar_ols(X, lag):
    p = X[:, :-lag] - X[:, :-lag].mean(1, keepdims=True)
    n = X[:, lag:] - X[:, lag:].mean(1, keepdims=True)
    A = np.linalg.solve((p @ p.T).T, (n @ p.T).T).T
    np.fill_diagonal(A, 0.0)
    return A


def infer_types(A):
    """Unsupervised: type of each source column = sign of its column sum."""
    t = np.sign(A.sum(0))
    t[t == 0] = 1.0
    return t


def type_accuracy(A):
    """Overall and inhibitory-only accuracy of the column-sign type guess."""
    t = infer_types(A)
    true = np.where(np.arange(A.shape[0]) < NE, 1.0, -1.0)
    overall = float(np.mean(t == true))
    inh = true < 0
    inh_acc = float(np.mean(t[inh] == true[inh]))
    return overall, inh_acc


def typeeq(A):
    """Unsupervised rescale: lift inferred-inhibitory group to match excitatory group."""
    N = A.shape[0]; off = ~np.eye(N, dtype=bool); B = A.copy()
    t = infer_types(A)
    Ecols = np.where(t > 0)[0]; Icols = np.where(t < 0)[0]
    maskE = np.zeros((N, N), bool); maskE[:, Ecols] = True; maskE &= off
    maskI = np.zeros((N, N), bool); maskI[:, Icols] = True; maskI &= off
    mE = np.median(np.abs(A[maskE])) if maskE.any() else 0.0
    mI = np.median(np.abs(A[maskI])) if maskI.any() else 0.0
    if mI > 1e-9 and mE > 0:
        B[:, Icols] = A[:, Icols] * (mE / mI)
    np.fill_diagonal(B, 0.0)
    return B


def oracle(A, adj):
    N = A.shape[0]; off = ~np.eye(N, dtype=bool); G = adj.T
    exc = off & (G > 0); inh = off & (G < 0)
    sE = (A[exc] * G[exc]).sum() / (G[exc] ** 2).sum()
    sI = (A[inh] * G[inh]).sum() / (G[inh] ** 2).sum()
    B = A.copy()
    for j in range(N):
        s = sE if j < NE else sI
        if abs(s) > 1e-12:
            B[:, j] = A[:, j] / s
    np.fill_diagonal(B, 0.0)
    return B


def main():
    adj = np.load(Path(DS) / "adj_true.npy"); np.fill_diagonal(adj, 0.0)
    feed = np.asarray(zarr.open(FEED, "r")[:])
    N = adj.shape[0]; off = ~np.eye(N, dtype=bool)
    m = connectivity_metrics._metrics

    def ei(A):
        g = adj.T[off]; a = np.abs(A[off]); inh, exc = a[g < 0], a[g > 0]
        return float(np.median(inh) / np.median(exc)) if len(inh) and len(exc) and np.median(exc) else float("nan")

    def show(A):
        return dict(spearman=m["spearman"](A, adj), pearson=m["pearson"](A, adj),
                    auc=m["auc_roc"](A, adj), f1=m["f1"](A, adj),
                    precision=m["precision"](A, adj), ei=ei(A))

    A_ols = ar_ols(feed, LAG)
    A_en = fista_solve(FEED, lag=LAG, lam_l1=3e-3, lam_l2=1e-3, n_iter=500, chunk_size=10000)
    np.fill_diagonal(A_en, 0.0)

    # (1) type-inference accuracy
    o_ols, i_ols = type_accuracy(A_ols)
    o_en, i_en = type_accuracy(A_en)
    print("=== type-inference accuracy (column-sign guess) ===")
    print(f"  raw OLS     : overall {o_ols:.2f}   inhibitory-only {i_ols:.2f}")
    print(f"  regularized : overall {o_en:.2f}   inhibitory-only {i_en:.2f}")

    # (2) end metrics
    methods = [("OLS", A_ols), ("EN3e-3", A_en),
               ("EN3e-3+typeeq(unsup)", typeeq(A_en)),
               ("EN3e-3+oracle(ceiling)", oracle(A_en, adj))]
    print("\n=== end metrics ===")
    print(f"{'method':>24}{'spearman':>10}{'pearson':>9}{'auc':>7}{'f1':>7}{'precision':>11}{'E/I':>7}")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    out_dir = Path(OUT); out_dir.mkdir(parents=True, exist_ok=True)
    for name, A in methods:
        r = show(A)
        print(f"{name:>24}{r['spearman']:>10.3f}{r['pearson']:>9.3f}{r['auc']:>7.3f}"
              f"{r['f1']:>7.3f}{r['precision']:>11.3f}{r['ei']:>7.2f}")
        cfg = ExperimentConfig(name=f"unsup/{name}", lag_ms=LAG * 0.1, dt=0.1,
                               n_excitatory=NE, n_inhibitory=N - NE, solver="fista",
                               output_dir=OUT, data_path=DS)
        res = ExperimentResult(config_path="-", loss_curve=[], duration_seconds=0.0, timestamp=ts,
                               run_dir=str(out_dir / name.replace("+", "_").replace("(", "").replace(")", "")),
                               spikes_path=None, calcium_path=None, feed_zarr_path=FEED,
                               adj_true_path=str(Path(DS) / "adj_true.npy"), adj_inferred_path="-",
                               metrics={f"connectivity/{k}": r[k] for k in ("spearman", "pearson", "auc", "f1", "precision")}
                                       | {"diag/ei_ratio": r["ei"]})
        append_row(res, cfg, out_dir / "ledger.csv")


if __name__ == "__main__":
    main()
