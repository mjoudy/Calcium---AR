"""
Regularization test — can Elastic Net clean up the exact connections (precision)?

The data-size test showed precision is VARIANCE-limited, so regularization should
help it.  Here we sweep the Elastic Net knobs (L1 = sparsity, L2 = stability) on
the real preprocessed feed at lag = 1.5 ms, and check whether precision / F1 rise
above the plain-OLS baseline WITHOUT crushing Pearson.

Watch for the classic Lasso trap: the true coefficients are tiny (median ≈ 0.005),
so a "normal" λ zeros everything.  We sweep λ over a wide log range to bracket the
sweet spot.

Uses the project's FISTA solver (calcium_ar/solvers/fista.py) on a feed zarr.
Every run is appended to the ledger.

Usage
-----
    python scripts/regularization_test.py
    python scripts/regularization_test.py --l1 1e-5,1e-4,1e-3,1e-2 --l2 1e-3,1e-1
"""

import argparse
import sys
from datetime import datetime, timezone
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
from calcium_ar.experiments.config import ExperimentConfig
from calcium_ar.experiments.result import ExperimentResult
from calcium_ar.experiments.ledger import append_row


def ar_ols(X, lag):
    p = X[:, :-lag] - X[:, :-lag].mean(1, keepdims=True)
    n = X[:, lag:]  - X[:, lag:].mean(1, keepdims=True)
    A = np.linalg.solve((p @ p.T).T, (n @ p.T).T).T
    np.fill_diagonal(A, 0.0)
    return A


def ei_ratio(A, adj):
    N = adj.shape[0]; m = ~np.eye(N, dtype=bool)
    g = adj.T[m]; a = np.abs(A[m])
    inh, exc = a[g < 0], a[g > 0]
    if len(inh) == 0 or len(exc) == 0 or np.median(exc) == 0:
        return float("nan")
    return float(np.median(inh) / np.median(exc))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="results/solver_comparison_N100/dataset")
    p.add_argument("--feed", default="results/regularization_test/feed.zarr")
    p.add_argument("--l1", default="0,1e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2")
    p.add_argument("--l2", default="1e-3,1e-1")
    p.add_argument("--lag", type=int, default=15)
    p.add_argument("--out", default="results/regularization_test")
    args = p.parse_args()

    ds = Path(args.dataset)
    adj = np.load(ds / "adj_true.npy"); np.fill_diagonal(adj, 0.0)
    feed = np.asarray(zarr.open(args.feed, "r")[:])
    N = adj.shape[0]
    dt = 0.1; lag = args.lag
    l1s = [float(x) for x in args.l1.split(",")]
    l2s = [float(x) for x in args.l2.split(",")]
    m = connectivity_metrics._metrics
    off = ~np.eye(N, dtype=bool)

    def metrics(A):
        return dict(pearson=m["pearson"](A, adj), precision=m["precision"](A, adj),
                    recall=m["recall"](A, adj), f1=m["f1"](A, adj),
                    auc=m["auc_roc"](A, adj), ei=ei_ratio(A, adj),
                    pct_zero=float(np.mean(np.abs(A[off]) < 1e-12)))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- OLS baseline ----
    base = metrics(ar_ols(feed, lag))
    print("baseline OLS  ->  " + "  ".join(f"{k}={base[k]:.3f}" for k in
          ("pearson", "precision", "recall", "f1", "auc", "ei")))

    def log(name, l1, l2, mv):
        cfg = ExperimentConfig(name=name, lag_ms=lag * dt, dt=dt, n_excitatory=80,
                               n_inhibitory=20, solver="fista", lam=l1, lam_l2=l2,
                               output_dir=args.out, data_path=str(ds))
        r = ExperimentResult(config_path="-", loss_curve=[], duration_seconds=0.0, timestamp=ts,
                             run_dir=str(out_dir / name.replace("/", "_")), spikes_path=None,
                             calcium_path=None, feed_zarr_path=args.feed,
                             adj_true_path=str(ds / "adj_true.npy"), adj_inferred_path="-",
                             metrics={"connectivity/pearson": mv["pearson"],
                                      "connectivity/precision": mv["precision"],
                                      "connectivity/recall": mv["recall"],
                                      "connectivity/f1": mv["f1"],
                                      "connectivity/auc_roc": mv["auc"],
                                      "diag/ei_ratio": mv["ei"], "diag/pct_zero": mv["pct_zero"]})
        append_row(r, cfg, Path(args.out) / "ledger.csv")

    log("reg/ols_baseline", 0.0, 0.0, base)

    # ---- Elastic Net grid ----
    grid = {}
    for l2 in l2s:
        print(f"\n--- L2 = {l2:g} ---")
        print(f"{'L1':>10}{'pearson':>9}{'precision':>11}{'recall':>8}{'f1':>7}{'auc':>7}{'ei':>7}{'%zero':>8}")
        for l1 in l1s:
            A = fista_solve(args.feed, lag=lag, lam_l1=l1, lam_l2=l2, n_iter=500, chunk_size=10000)
            np.fill_diagonal(A, 0.0)
            mv = metrics(A)
            grid[(l1, l2)] = mv
            print(f"{l1:>10.0e}{mv['pearson']:>9.3f}{mv['precision']:>11.3f}"
                  f"{mv['recall']:>8.3f}{mv['f1']:>7.3f}{mv['auc']:>7.3f}{mv['ei']:>7.2f}"
                  f"{100*mv['pct_zero']:>7.0f}%")
            log(f"reg/en_l1{l1:g}_l2{l2:g}", l1, l2, mv)

    # ---- plot precision & f1 vs L1 ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    x = [max(l, 1e-6) for l in l1s]
    for l2 in l2s:
        ax[0].plot(x, [grid[(l1, l2)]["precision"] for l1 in l1s], "o-", label=f"L2={l2:g}")
        ax[1].plot(x, [grid[(l1, l2)]["f1"] for l1 in l1s], "o-", label=f"L2={l2:g}")
    ax[0].axhline(base["precision"], color="k", ls="--", lw=0.8, label="OLS baseline")
    ax[1].axhline(base["f1"], color="k", ls="--", lw=0.8, label="OLS baseline")
    ax[0].set(xscale="log", xlabel="L1 (lam_l1)", ylabel="precision", title="Precision vs L1")
    ax[1].set(xscale="log", xlabel="L1 (lam_l1)", ylabel="F1", title="F1 vs L1")
    for a in ax:
        a.legend(); a.grid(alpha=0.3)
    fig.suptitle(f"Regularization (Elastic Net) — lag={lag} samples ({lag*dt} ms)")
    fig.tight_layout(); fig.savefig(out_dir / "regularization_test.png", dpi=150)
    print(f"\nplot → {out_dir/'regularization_test.png'}")


if __name__ == "__main__":
    main()
