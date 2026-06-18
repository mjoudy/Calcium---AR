"""
Data-size test — is the ceiling caused by too little data (variance) or by the
method itself (bias)?

Same existing dataset, fixed lag = synaptic delay (1.5 ms = 15 samples), but the
recording is truncated to increasing lengths.  We watch whether recovery keeps
climbing with more data (variance-limited → more data / regularization helps) or
flattens (bias-limited → neither helps; needs post-processing).

Two inputs:
    spikes  — perfect spikes (the ceiling)
    feed    — your real preprocessing on noisy calcium (the real pipeline)

Headline metric: Pearson.  Precision/recall/F1/AUC and the E/I ratio alongside.
Every (input, length) run is appended to the ledger (sim_time encodes the length).

Usage
-----
    python scripts/data_size_test.py
    python scripts/data_size_test.py --fracs 2500,5000,10000,20000,35000,50000
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calcium_ar.simulation import simulate_calcium
from calcium_ar.preprocessing.signal_utils import get_signal_derivative_pair
from calcium_ar.preprocessing.tau_estimation import estimate_tau_robust
from calcium_ar.preprocessing.feed_reconstruction import reconstruct_feed
from calcium_ar.experiments.metrics import connectivity_metrics
from calcium_ar.experiments.config import ExperimentConfig
from calcium_ar.experiments.result import ExperimentResult
from calcium_ar.experiments.ledger import append_row


def ar_ols(X, lag):
    p = X[:, :-lag] - X[:, :-lag].mean(1, keepdims=True)
    n = X[:, lag:]  - X[:, lag:].mean(1, keepdims=True)
    C_xx = p @ p.T            # (N, N), symmetric
    C_yx = n @ p.T            # (N, N)
    # lstsq (pseudo-inverse) instead of solve: robust when a neuron is too
    # sparse in a short window and C_xx is singular. A = C_yx @ inv(C_xx).
    sol, *_ = np.linalg.lstsq(C_xx, C_yx.T, rcond=None)
    A = sol.T
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
    p.add_argument("--fracs", default="2500,5000,10000,20000,35000,50000",
                   help="comma-separated recording lengths in SAMPLES")
    p.add_argument("--lag", type=int, default=15, help="AR lag in samples (15 = 1.5 ms)")
    p.add_argument("--out", default="results/data_size_test")
    args = p.parse_args()

    import zarr
    ds = Path(args.dataset)
    meta = json.loads((ds / "metadata.json").read_text())["sim_params"]
    dt = meta["dt"]; NE = meta["n_excitatory"]
    spikes = np.asarray(zarr.open(str(ds / "spikes.zarr"), "r")[:])
    adj = np.load(ds / "adj_true.npy"); np.fill_diagonal(adj, 0.0)
    N, T = spikes.shape
    lag = args.lag
    lengths = [int(x) for x in args.fracs.split(",") if int(x) <= T]

    # build the real preprocessed feed once, then slice
    cal = simulate_calcium(spikes, tau=meta["tau"], dt=dt, amplitude=meta["amplitude"],
                           sigma_intra=meta["sigma_intra"], sigma_extra=meta["sigma_extra"], seed=2)
    sm, dv = get_signal_derivative_pair(cal, window_length=5, delta=dt)
    te = estimate_tau_robust(cal, window_length=5, method=meta["tau_method"], dt=dt)
    feed = reconstruct_feed(sm, dv, te)

    inputs = {"spikes": spikes.astype(float), "feed": feed}
    m = connectivity_metrics._metrics
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    res = {s: {k: [] for k in ("T", "pearson", "precision", "auc", "ei")} for s in inputs}

    for stage, X in inputs.items():
        for L in lengths:
            A = ar_ols(X[:, :L], lag)
            pe = m["pearson"](A, adj); pr = m["precision"](A, adj)
            au = m["auc_roc"](A, adj); ei = ei_ratio(A, adj)
            res[stage]["T"].append(L); res[stage]["pearson"].append(pe)
            res[stage]["precision"].append(pr); res[stage]["auc"].append(au); res[stage]["ei"].append(ei)

            cfg = ExperimentConfig(name=f"datasize/{stage}", lag_ms=lag * dt, dt=dt,
                                   sim_time=L * dt, n_excitatory=NE, n_inhibitory=N - NE,
                                   tau=meta["tau"], solver="ols", output_dir=args.out, data_path=str(ds))
            r = ExperimentResult(config_path="-", loss_curve=[], duration_seconds=0.0, timestamp=ts,
                                 run_dir=str(out_dir / f"{stage}_T{L}"), spikes_path=None, calcium_path=None,
                                 feed_zarr_path="-", adj_true_path=str(ds / "adj_true.npy"), adj_inferred_path="-",
                                 metrics={"connectivity/pearson": pe, "connectivity/precision": pr,
                                          "connectivity/auc_roc": au, "diag/ei_ratio": ei})
            append_row(r, cfg, Path(args.out) / "ledger.csv")

    # ---- print ----
    def show(metric, label):
        print(f"\n=== {label} — rows: input, cols: recording length (samples) ===")
        print("input".ljust(10) + "".join(f"{L:>9d}" for L in lengths))
        for s in inputs:
            print(s.ljust(10) + "".join(f"{v:>9.3f}" for v in res[s][metric]))
    show("pearson", "Pearson (headline)")
    show("precision", "Precision (exact connections)")
    show("ei", "E/I ratio (truth = 5.0)")

    # ---- plot ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for s in inputs:
        ax[0].plot(res[s]["T"], res[s]["pearson"], "o-", label=s)
        ax[1].plot(res[s]["T"], res[s]["precision"], "o-", label=s)
    ax[0].set(xlabel="recording length (samples)", ylabel="Pearson", title="Pearson vs data size")
    ax[1].set(xlabel="recording length (samples)", ylabel="Precision", title="Precision vs data size")
    for a in ax:
        a.legend(); a.grid(alpha=0.3)
    fig.suptitle(f"Data-size test — N={N}, lag={lag} samples ({lag*dt} ms)")
    fig.tight_layout(); fig.savefig(out_dir / "data_size_test.png", dpi=150)
    print(f"\nplot → {out_dir/'data_size_test.png'}")


if __name__ == "__main__":
    main()
