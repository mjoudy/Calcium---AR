"""
Oracle-ladder diagnostic — where does the connectivity signal die?

Runs the SAME AR connectivity estimator on progressively more corrupted inputs,
all derived from one existing simulated dataset (no new network, no ground-truth
changes), and reports how well each recovers the ground truth.

Rungs (input to the AR regression)
----------------------------------
    spikes        true spike trains            — the ceiling
    clean_calcium calcium with NO noise → your preprocessing → feed
    noisy_calcium calcium with real noise → your preprocessing → feed  (real pipeline)

Reading the gaps
----------------
    spikes vs clean   = cost of the calcium low-pass + feed reconstruction
    clean vs noisy    = cost of recording noise (attenuation)
    if spikes is already poor → the regression itself is the bottleneck,
                                calcium is innocent.

Headline metric is Pearson correlation (off-diagonal, adj_true.T convention).
AUC and the E/I magnitude ratio are reported alongside.  Every (rung, lag) run
is appended to the experiment ledger.

Usage
-----
    python scripts/oracle_ladder.py
    python scripts/oracle_ladder.py --dataset results/solver_comparison_N100/dataset
    python scripts/oracle_ladder.py --lags 5,10,15,20,30,50
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


# --------------------------------------------------------------------------- #
# AR estimator — in-memory, identical math to solvers/chunked_ols.py
# (centred OLS: A = C_yx_c @ inv(C_xx_c), diagonal zeroed)
# --------------------------------------------------------------------------- #

def ar_ols(X: np.ndarray, lag: int) -> np.ndarray:
    """Estimate A from X_t = A X_{t-lag} with a free intercept (mean-centring)."""
    x_prev = X[:, :-lag]
    x_now  = X[:, lag:]
    x_prev = x_prev - x_prev.mean(axis=1, keepdims=True)
    x_now  = x_now  - x_now.mean(axis=1, keepdims=True)
    C_xx = x_prev @ x_prev.T
    C_yx = x_now  @ x_prev.T
    A = np.linalg.solve(C_xx.T, C_yx.T).T
    np.fill_diagonal(A, 0.0)
    return A


def ei_ratio(adj_inferred: np.ndarray, adj_true: np.ndarray) -> float:
    """median|Â| over true-inhibitory edges  /  over true-excitatory edges.

    Ground-truth ratio is g (=5 here, |J_in|/|J_ex|=50/10).  Tells us whether the
    estimate preserves the E/I magnitude relationship or compresses it toward 1.
    """
    N = adj_true.shape[0]
    mask = ~np.eye(N, dtype=bool)
    g = adj_true.T[mask]
    a = np.abs(adj_inferred[mask])
    inh, exc = a[g < 0], a[g > 0]
    if len(inh) == 0 or len(exc) == 0 or np.median(exc) == 0:
        return float("nan")
    return float(np.median(inh) / np.median(exc))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="results/solver_comparison_N100/dataset")
    p.add_argument("--lags", default="5,10,15,20,30,50",
                   help="comma-separated AR lags in SAMPLES")
    p.add_argument("--smooth-window", type=int, default=5,
                   help="Savitzky-Golay window in samples (preprocessing)")
    p.add_argument("--out", default="results/oracle_ladder")
    args = p.parse_args()

    import zarr
    ds = Path(args.dataset)
    meta = json.loads((ds / "metadata.json").read_text())["sim_params"]
    dt          = meta["dt"]
    tau_true    = meta["tau"]
    amplitude   = meta["amplitude"]
    sigma_intra = meta["sigma_intra"]
    sigma_extra = meta["sigma_extra"]
    tau_method  = meta["tau_method"]
    NE          = meta["n_excitatory"]

    spikes   = np.asarray(zarr.open(str(ds / "spikes.zarr"), "r")[:])   # (N, T)
    adj_true = np.load(ds / "adj_true.npy")
    np.fill_diagonal(adj_true, 0.0)
    N, T = spikes.shape
    lags = [int(x) for x in args.lags.split(",")]
    win  = args.smooth_window

    print(f"dataset: N={N}  T={T}  dt={dt}ms  tau={tau_true}ms  "
          f"NE={NE}  noise(extra)={sigma_extra}")
    print(f"synaptic delay ~1.5ms = {round(1.5/dt)} samples; lags(samples)={lags}\n")

    # ---- Build the three input signals (reuse same spikes — no new network) ---
    def feed_from_calcium(sig_intra, sig_extra, seed):
        cal = simulate_calcium(spikes, tau=tau_true, dt=dt, amplitude=amplitude,
                               sigma_intra=sig_intra, sigma_extra=sig_extra, seed=seed)
        smooth, deriv = get_signal_derivative_pair(cal, window_length=win, delta=dt)
        tau_est = estimate_tau_robust(cal, window_length=win, method=tau_method, dt=dt)
        return reconstruct_feed(smooth, deriv, tau_est)

    print("building inputs …")
    inputs = {
        "spikes":        spikes.astype(float),
        "clean_calcium": feed_from_calcium(0.0,         0.0,         seed=1),
        "noisy_calcium": feed_from_calcium(sigma_intra, sigma_extra, seed=2),
    }

    # ---- Run the grid: rung x lag --------------------------------------------
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    results = {stage: {"lag_ms": [], "pearson": [], "auc": [], "ei": []}
               for stage in inputs}

    pearson_fn = connectivity_metrics._metrics["pearson"]
    auc_fn     = connectivity_metrics._metrics["auc_roc"]

    for stage, X in inputs.items():
        for lag in lags:
            A = ar_ols(X, lag)
            pe  = pearson_fn(A, adj_true)
            auc = auc_fn(A, adj_true)
            ei  = ei_ratio(A, adj_true)
            results[stage]["lag_ms"].append(lag * dt)
            results[stage]["pearson"].append(pe)
            results[stage]["auc"].append(auc)
            results[stage]["ei"].append(ei)

            # log to ledger
            cfg = ExperimentConfig(
                name=f"ladder/{stage}", lag_ms=lag * dt, dt=dt,
                n_excitatory=NE, n_inhibitory=N - NE, tau=tau_true,
                sigma_extra=(0.0 if stage == "clean_calcium" else sigma_extra),
                smooth_window_ms=win * dt, solver="ols",
                output_dir=args.out, data_path=str(ds),
            )
            res = ExperimentResult(
                config_path="-", loss_curve=[], duration_seconds=0.0, timestamp=ts,
                run_dir=str(out_dir / f"{stage}_lag{lag}"),
                spikes_path=None, calcium_path=None, feed_zarr_path="-",
                adj_true_path=str(ds / "adj_true.npy"), adj_inferred_path="-",
                metrics={"connectivity/pearson": pe,
                         "connectivity/auc_roc": auc,
                         "diag/ei_ratio": ei},
            )
            append_row(res, cfg, Path(args.out) / "ledger.csv")

    # ---- Print summary table -------------------------------------------------
    print("\n=== Pearson (headline) — rows: rung, cols: lag(ms) ===")
    header = "rung".ljust(15) + "".join(f"{l*dt:>8.1f}" for l in lags)
    print(header); print("-" * len(header))
    for stage in inputs:
        row = stage.ljust(15) + "".join(f"{v:>8.3f}" for v in results[stage]["pearson"])
        print(row)

    print("\n=== AUC-ROC ===")
    for stage in inputs:
        print(stage.ljust(15) + "".join(f"{v:>8.3f}" for v in results[stage]["auc"]))

    print("\n=== E/I magnitude ratio (ground truth = 5.0) ===")
    for stage in inputs:
        print(stage.ljust(15) + "".join(f"{v:>8.2f}" for v in results[stage]["ei"]))

    # ---- Plot ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for stage in inputs:
        r = results[stage]
        axes[0].plot(r["lag_ms"], r["pearson"], "o-", label=stage)
        axes[1].plot(r["lag_ms"], r["auc"], "o-", label=stage)
    axes[0].axvline(1.5, color="k", ls="--", lw=0.8, label="synaptic delay 1.5ms")
    axes[0].set(xlabel="lag (ms)", ylabel="Pearson", title="Headline: Pearson vs lag")
    axes[1].axvline(1.5, color="k", ls="--", lw=0.8)
    axes[1].set(xlabel="lag (ms)", ylabel="AUC-ROC", title="AUC vs lag")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"Oracle ladder — N={N}, T={T}, tau={tau_true}ms")
    fig.tight_layout()
    fig.savefig(out_dir / "oracle_ladder.png", dpi=150)
    print(f"\nplot → {out_dir/'oracle_ladder.png'}")
    print(f"ledger → {Path(args.out)/'ledger.csv'}")


if __name__ == "__main__":
    main()
