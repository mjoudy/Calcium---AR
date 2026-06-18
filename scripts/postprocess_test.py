"""
Post-processing test — can rescaling undo the inhibition bias?

The data-size test showed the E/I magnitude is BIAS-limited (regularization can't
fix it).  Here we try post-processing the OLS estimate (lag = 1.5 ms) to un-shrink
inhibition, and check Spearman / AUC / F1 / E-I ratio.

Methods
-------
    baseline    raw OLS, no post-processing
    colnorm     UNSUPERVISED: divide each neuron's column by its own std
                (cancels the per-neuron hidden scale; pulls weak inhibition back up)
    typeeq      UNSUPERVISED: guess each neuron's type from its column sign, then
                scale the inferred-inhibitory group up to match the excitatory group
    oracle      CEILING: use TRUE types + per-class slope to map A back onto G's scale
                (best a pure rescaling could ever do — not a real method)

Headline metric is Spearman (rank agreement — the right measure for a nonlinear,
discrete-weight map).  Every method is appended to the ledger.

Usage
-----
    python scripts/postprocess_test.py
"""

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

from calcium_ar.experiments.metrics import connectivity_metrics
from calcium_ar.experiments.config import ExperimentConfig
from calcium_ar.experiments.result import ExperimentResult
from calcium_ar.experiments.ledger import append_row

DS   = "results/solver_comparison_N100/dataset"
FEED = "results/regularization_test/feed.zarr"
OUT  = "results/postprocess_test"
NE   = 80
LAG  = 15


def ar_ols(X, lag):
    p = X[:, :-lag] - X[:, :-lag].mean(1, keepdims=True)
    n = X[:, lag:]  - X[:, lag:].mean(1, keepdims=True)
    A = np.linalg.solve((p @ p.T).T, (n @ p.T).T).T
    np.fill_diagonal(A, 0.0)
    return A


# --------------------------------------------------------------------------- #
# Post-processing methods (each takes A, returns a new A)
# --------------------------------------------------------------------------- #

def colnorm(A):
    """Unsupervised: divide each column by its off-diagonal std."""
    N = A.shape[0]; B = A.copy()
    for j in range(N):
        col = B[:, j]; sd = col[np.arange(N) != j].std()
        if sd > 1e-9:
            B[:, j] = col / sd
    np.fill_diagonal(B, 0.0)
    return B


def typeeq(A):
    """Unsupervised: infer type from column sign, lift inferred-I group to match E."""
    N = A.shape[0]; B = A.copy(); off = ~np.eye(N, dtype=bool)
    tj = np.sign(A.sum(0))                       # inferred type per source column
    Ecols = np.where(tj > 0)[0]; Icols = np.where(tj < 0)[0]
    # median |off-diag| within each inferred group
    maskE = np.zeros((N, N), bool); maskE[:, Ecols] = True; maskE &= off
    maskI = np.zeros((N, N), bool); maskI[:, Icols] = True; maskI &= off
    mE = np.median(np.abs(A[maskE])) if maskE.any() else 0.0
    mI = np.median(np.abs(A[maskI])) if maskI.any() else 0.0
    if mI > 1e-9:
        B[:, Icols] = A[:, Icols] * (mE / mI)
    np.fill_diagonal(B, 0.0)
    return B


def oracle(A, adj):
    """Ceiling: true types + per-class least-squares slope mapping A onto G."""
    N = A.shape[0]; off = ~np.eye(N, dtype=bool)
    G = adj.T
    exc = off & (G > 0); inh = off & (G < 0)
    sE = (A[exc] * G[exc]).sum() / (G[exc] ** 2).sum()
    sI = (A[inh] * G[inh]).sum() / (G[inh] ** 2).sum()
    B = A.copy()
    for j in range(N):
        s = sE if j < NE else sI            # true source type by index
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

    A = ar_ols(feed, LAG)
    methods = {"baseline": A, "colnorm": colnorm(A), "typeeq": typeeq(A), "oracle": oracle(A, adj)}

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    out_dir = Path(OUT); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'method':>10}{'spearman':>10}{'pearson':>9}{'auc':>7}{'precision':>11}{'recall':>8}{'f1':>7}{'E/I':>7}")
    rows = {}
    for name, B in methods.items():
        r = dict(spearman=m["spearman"](B, adj), pearson=m["pearson"](B, adj),
                 auc=m["auc_roc"](B, adj), precision=m["precision"](B, adj),
                 recall=m["recall"](B, adj), f1=m["f1"](B, adj), ei=ei(B))
        rows[name] = r
        print(f"{name:>10}{r['spearman']:>10.3f}{r['pearson']:>9.3f}{r['auc']:>7.3f}"
              f"{r['precision']:>11.3f}{r['recall']:>8.3f}{r['f1']:>7.3f}{r['ei']:>7.2f}")
        cfg = ExperimentConfig(name=f"postproc/{name}", lag_ms=LAG * 0.1, dt=0.1,
                               n_excitatory=NE, n_inhibitory=N - NE, solver="ols",
                               output_dir=OUT, data_path=DS)
        res = ExperimentResult(config_path="-", loss_curve=[], duration_seconds=0.0, timestamp=ts,
                               run_dir=str(out_dir / name), spikes_path=None, calcium_path=None,
                               feed_zarr_path=FEED, adj_true_path=str(Path(DS) / "adj_true.npy"),
                               adj_inferred_path="-",
                               metrics={"connectivity/spearman": r["spearman"],
                                        "connectivity/pearson": r["pearson"],
                                        "connectivity/auc_roc": r["auc"],
                                        "connectivity/precision": r["precision"],
                                        "connectivity/recall": r["recall"],
                                        "connectivity/f1": r["f1"], "diag/ei_ratio": r["ei"]})
        append_row(res, cfg, out_dir / "ledger.csv")

    # bar chart
    names = list(methods); x = np.arange(len(names))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for a, key, ttl in zip(ax, ["spearman", "f1", "ei"],
                           ["Spearman", "F1", "E/I ratio (truth=5)"]):
        a.bar(x, [rows[n][key] for n in names], color=["gray", "steelblue", "tomato", "green"])
        a.set_xticks(x); a.set_xticklabels(names, rotation=15); a.set_title(ttl); a.grid(alpha=0.3, axis="y")
    ax[2].axhline(5.0, color="k", ls="--", lw=0.8, label="true ratio = 5")
    ax[2].legend()
    fig.suptitle("Post-processing the OLS estimate (lag = 1.5 ms)")
    fig.tight_layout(); fig.savefig(out_dir / "postprocess_test.png", dpi=150)
    print(f"\nplot → {out_dir/'postprocess_test.png'}")


if __name__ == "__main__":
    main()
