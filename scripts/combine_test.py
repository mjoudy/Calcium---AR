"""
Regularize-then-rescale — does the order break the detection/magnitude trade-off?

Regularization helps detection but zeros inhibition; rescaling helps magnitude but
amplifies noise.  Hypothesis: regularize FIRST (zero the noise), THEN rescale (lift
only the surviving real signal) — getting both.

Compares, at lag = 1.5 ms:
    OLS                      baseline
    OLS + oracle rescale     rescale only (from postprocess_test)
    EN(L1)                   regularize only
    EN(L1) + oracle          regularize then rescale (true types — ceiling)
    EN(L1) + colnorm         regularize then rescale (unsupervised)

over L1 in {1e-3, 3e-3, 1e-2}.  Headline = Spearman; also F1 (detection) and E/I.
Logs every method to the ledger.

Usage
-----
    python scripts/combine_test.py
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

DS   = "results/solver_comparison_N100/dataset"
FEED = "results/regularization_test/feed.zarr"
OUT  = "results/combine_test"
NE   = 80
LAG  = 15


def ar_ols(X, lag):
    p = X[:, :-lag] - X[:, :-lag].mean(1, keepdims=True)
    n = X[:, lag:]  - X[:, lag:].mean(1, keepdims=True)
    A = np.linalg.solve((p @ p.T).T, (n @ p.T).T).T
    np.fill_diagonal(A, 0.0)
    return A


def colnorm(A):
    N = A.shape[0]; B = A.copy()
    for j in range(N):
        col = B[:, j]; sd = col[np.arange(N) != j].std()
        if sd > 1e-9:
            B[:, j] = col / sd
    np.fill_diagonal(B, 0.0)
    return B


def oracle(A, adj):
    N = A.shape[0]; off = ~np.eye(N, dtype=bool); G = adj.T
    exc = off & (G > 0); inh = off & (G < 0)
    dE = (G[exc] ** 2).sum(); dI = (G[inh] ** 2).sum()
    sE = (A[exc] * G[exc]).sum() / dE if dE else 1.0
    sI = (A[inh] * G[inh]).sum() / dI if dI else 1.0
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

    def row(A):
        return dict(spearman=m["spearman"](A, adj), pearson=m["pearson"](A, adj),
                    auc=m["auc_roc"](A, adj), f1=m["f1"](A, adj),
                    precision=m["precision"](A, adj), recall=m["recall"](A, adj), ei=ei(A))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    out_dir = Path(OUT); out_dir.mkdir(parents=True, exist_ok=True)

    A_ols = ar_ols(feed, LAG)
    methods = [("OLS", A_ols),
               ("OLS+oracle", oracle(A_ols, adj))]
    for l1 in (1e-3, 3e-3, 1e-2):
        A = fista_solve(FEED, lag=LAG, lam_l1=l1, lam_l2=1e-3, n_iter=500, chunk_size=10000)
        np.fill_diagonal(A, 0.0)
        methods.append((f"EN{l1:g}", A))
        methods.append((f"EN{l1:g}+oracle", oracle(A, adj)))
        methods.append((f"EN{l1:g}+colnorm", colnorm(A)))

    print(f"{'method':>16}{'spearman':>10}{'pearson':>9}{'auc':>7}{'f1':>7}{'precision':>11}{'E/I':>7}")
    for name, A in methods:
        r = row(A)
        print(f"{name:>16}{r['spearman']:>10.3f}{r['pearson']:>9.3f}{r['auc']:>7.3f}"
              f"{r['f1']:>7.3f}{r['precision']:>11.3f}{r['ei']:>7.2f}")
        cfg = ExperimentConfig(name=f"combo/{name}", lag_ms=LAG * 0.1, dt=0.1,
                               n_excitatory=NE, n_inhibitory=N - NE, solver="fista",
                               output_dir=OUT, data_path=DS)
        res = ExperimentResult(config_path="-", loss_curve=[], duration_seconds=0.0, timestamp=ts,
                               run_dir=str(out_dir / name.replace("+", "_")), spikes_path=None,
                               calcium_path=None, feed_zarr_path=FEED,
                               adj_true_path=str(Path(DS) / "adj_true.npy"), adj_inferred_path="-",
                               metrics={f"connectivity/{k}": r[k] for k in
                                        ("spearman", "pearson", "auc", "f1", "precision", "recall")}
                                       | {"diag/ei_ratio": r["ei"]})
        append_row(res, cfg, out_dir / "ledger.csv")


if __name__ == "__main__":
    main()
