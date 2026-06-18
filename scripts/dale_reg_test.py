"""
Dale-as-regularization test — enforce single-sign columns DURING the fit.

Compares the in-solver Dale constraint against the post-hoc Dale cleanup, to test the
theoretical claim that constraining during the fit (so correct-sign entries re-grow to
explain the variance) beats deleting wrong-sign entries afterwards.

Method (sign-constrained Elastic Net, solved with FISTA):
    minimise (1/2T)||X_t - A X_lag||^2 + L1||A||_1 + (L2/2)||A||^2
    subject to  sign(A[i,j]) = t_j   (each column single-signed)
Implemented by adding a per-column sign projection to the FISTA proximal step.
Types t_j: strongest-entry rule on an initial EN fit (unsupervised), or TRUE (ceiling).

Usage:  python scripts/dale_reg_test.py
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
OUT = "results/dale_reg_test"
NE = 80
LAG = 15


def strongest_entry_types(A):
    N = A.shape[0]; t = np.ones(N)
    for j in range(N):
        c = A[:, j].copy(); c[j] = 0.0
        t[j] = np.sign(c[np.argmax(np.abs(c))]) or 1.0
    return t


def dale_cleanup(A, t):
    B = A.copy()
    for j in range(A.shape[0]):
        col = B[:, j]; col[np.sign(col) != t[j]] = 0.0; B[:, j] = col
    np.fill_diagonal(B, 0.0)
    return B


def dale_fista(X, Y, types, lam1, lam2, n_iter=800):
    """Sign-constrained Elastic Net via FISTA. X,Y are centred (N, M)."""
    N, M = X.shape
    Cxx = (X @ X.T) / M
    Cyx = (Y @ X.T) / M
    Lip = np.linalg.eigvalsh(Cxx)[-1] + lam2
    thr = lam1 / Lip
    tcol = types[None, :]                       # (1,N) — column j sign
    A = np.zeros((N, N)); Z = A.copy(); tk = 1.0
    for _ in range(n_iter):
        grad = Z @ Cxx - Cyx + lam2 * Z
        V = Z - grad / Lip
        V = np.sign(V) * np.maximum(np.abs(V) - thr, 0.0)   # soft-threshold (L1)
        V = tcol * np.maximum(tcol * V, 0.0)                # per-column sign projection
        tnew = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        Z = V + ((tk - 1) / tnew) * (V - A)
        A, tk = V, tnew
    np.fill_diagonal(A, 0.0)
    return A


def main():
    adj = np.load(Path(DS) / "adj_true.npy"); np.fill_diagonal(adj, 0.0)
    feed = np.asarray(zarr.open(FEED, "r")[:])
    N = adj.shape[0]; off = ~np.eye(N, dtype=bool)
    m = connectivity_metrics._metrics

    def ei(A):
        g = adj.T[off]; a = np.abs(A[off]); inh, exc = a[g < 0], a[g > 0]
        return float(np.median(inh) / np.median(exc)) if len(inh) and len(exc) and np.median(exc) else float("nan")

    def score(A):
        return dict(spearman=m["spearman"](A, adj), pearson=m["pearson"](A, adj),
                    auc=m["auc_roc"](A, adj), f1=m["f1"](A, adj),
                    precision=m["precision"](A, adj), type_acc=m["dale_type_accuracy"](A, adj),
                    ei=ei(A), dale=m["degree_of_daleianity"](A, adj))

    # centred lag pairs (same convention as chunked_ols)
    X = feed[:, :-LAG] - feed[:, :-LAG].mean(1, keepdims=True)
    Y = feed[:, LAG:] - feed[:, LAG:].mean(1, keepdims=True)

    A_en = fista_solve(FEED, lag=LAG, lam_l1=3e-3, lam_l2=1e-3, n_iter=500, chunk_size=10000)
    np.fill_diagonal(A_en, 0.0)
    t_unsup = strongest_entry_types(A_en)
    t_true = np.where(np.arange(N) < NE, 1.0, -1.0)

    methods = [
        ("EN(3e-3)",                "regularize", False, A_en),
        ("EN+dale_postproc",        "reg+dale",   False, dale_cleanup(A_en, t_unsup)),
        ("EN_daleReg(unsup types)", "dale-reg",   False, dale_fista(X, Y, t_unsup, 3e-3, 1e-3)),
        ("EN_daleReg(true types)",  "dale-reg*",  True,  dale_fista(X, Y, t_true, 3e-3, 1e-3)),
    ]

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    out_dir = Path(OUT); out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["spearman", "pearson", "auc", "f1", "precision", "type_acc", "ei", "dale"]
    print(f"{'method':>26}{'GT':>4}" + "".join(f"{c[:8]:>9}" for c in cols))
    print("-" * (30 + 9 * len(cols)))
    for name, kind, gt, A in methods:
        s = score(A)
        print(f"{name:>26}{('Y' if gt else '·'):>4}" + "".join(f"{s[c]:>9.3f}" for c in cols))
        cfg = ExperimentConfig(name=f"dalereg/{name}", lag_ms=LAG * 0.1, dt=0.1,
                               n_excitatory=NE, n_inhibitory=N - NE, solver="fista",
                               output_dir=OUT, data_path=DS)
        res = ExperimentResult(config_path="-", loss_curve=[], duration_seconds=0.0, timestamp=ts,
                               run_dir=str(out_dir / name.replace("(", "").replace(")", "").replace(" ", "_")),
                               spikes_path=None, calcium_path=None, feed_zarr_path=FEED,
                               adj_true_path=str(Path(DS) / "adj_true.npy"), adj_inferred_path="-",
                               metrics={f"connectivity/{k}": s[k] for k in ("spearman", "pearson", "auc", "f1", "precision")}
                                       | {"diag/ei_ratio": s["ei"], "diag/daleianity": s["dale"], "diag/type_acc": s["type_acc"]})
        append_row(res, cfg, out_dir / "ledger.csv")


if __name__ == "__main__":
    main()
