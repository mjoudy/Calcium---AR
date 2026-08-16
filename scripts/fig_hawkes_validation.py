"""
Pernice et al. 2011 Figure 2-style validation panel for our Hawkes ground
truth: raster of a subset of neurons, measured-vs-predicted rate scatter, and
measured-vs-predicted correlation scatter. Same purpose as their own Fig. 2 --
demonstrate that the simulated point process actually matches its own
closed-form linear prediction (Eqs. 9-10 in the paper).

"Predicted" comes from the exact linear formulas (B=(1-G)^-1, y=B*y0,
C=B*Y*B^T), rebuilt from the same G/margin/rate0 used to generate the run
(read from meta.npy). "Measured" rate is just spike count / duration; measured
correlation uses windowed spike-count covariance (bin_ms, matching the paper's
own Eq. 11 definition -- a large-window count covariance, not a single-lag
regression).

Usage:
  python scripts/fig_hawkes_validation.py \\
      --events ~/calcium_results/hawkes_events/n1250_tuned \\
      --out figures/fig_hawkes_validation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs
from hawkes_ground_truth import build_G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="~/calcium_results/hawkes_events/n1250_tuned")
    ap.add_argument("--out", default="figures/fig_hawkes_validation")
    ap.add_argument("--raster-neurons", type=int, default=60)
    ap.add_argument("--raster-window-ms", type=float, default=1500.0)
    ap.add_argument("--corr-bin-ms", type=float, default=50.0)
    args = ap.parse_args()

    d = Path(args.events).expanduser()
    idx = np.load(d / "idx.npy").astype(np.int64)
    times = np.load(d / "times_ms.npy").astype(np.float64)
    adj = np.load(d / "adj_true.npy").astype(np.float64)
    meta = np.load(d / "meta.npy", allow_pickle=True).item()
    np.fill_diagonal(adj, 0.0)
    N = adj.shape[0]
    sim_time = float(meta["sim_time"])
    rate0 = float(meta["rate0"])
    margin = float(meta["margin"])
    print(f"N={N}  sim_time={sim_time:.0f}ms  rate0={rate0:.2f}Hz  margin={margin}", flush=True)

    # --- rebuild the exact same G used to generate this run ----------------- #
    G, ginfo = build_G(adj, margin)
    B = np.linalg.inv(np.eye(N) - G)
    y0_vec = np.full(N, rate0)
    y_theory = B @ y0_vec                                 # Eq. 10, Hz
    C_theory = B @ np.diag(y_theory) @ B.T                 # Eq. 13 (y0 normalised
                                                             # away -> use y_theory
                                                             # directly as Y)

    # --- measured rate -------------------------------------------------------#
    dur_s = sim_time / 1000.0
    y_meas = np.bincount(idx, minlength=N) / dur_s

    # --- measured correlation: windowed spike-count covariance (Eq. 11) ----- #
    bin_ms = args.corr_bin_ms
    n_bins = int(sim_time // bin_ms)
    b = np.clip((times / bin_ms).astype(np.int64), 0, n_bins - 1)
    flat = idx * n_bins + b
    counts = np.bincount(flat, minlength=N * n_bins).reshape(N, n_bins).astype(np.float64)
    Xc = counts - counts.mean(1, keepdims=True)
    # Eq. 11: c_ij = cov(n_i(D), n_j(D)) / D  (D in ms here, matches C_theory's
    # per-ms rate units)
    C_meas = (Xc @ Xc.T) / n_bins / bin_ms

    off = ~np.eye(N, dtype=bool)

    fs.apply_style()
    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.35, wspace=0.28)

    # A: raster -----------------------------------------------------------------
    axA = fig.add_subplot(gs[0, :])
    rng = np.random.default_rng(0)
    neurons = rng.choice(N, size=min(args.raster_neurons, N), replace=False)
    keep_t = times < args.raster_window_ms
    pos = {n: i for i, n in enumerate(neurons)}
    sel = keep_t & np.isin(idx, neurons)
    ys = np.array([pos[k] for k in idx[sel]])
    axA.scatter(times[sel], ys, s=2, color=fs.ACCENT, alpha=0.7, linewidths=0)
    axA.set(xlabel="time [ms]", ylabel="neuron (subset)",
            title=f"A: raster, {len(neurons)} neurons (mean rate {y_meas.mean():.1f} Hz)")
    fs.despine(axA)

    # C: measured vs predicted rate ---------------------------------------------
    axC = fig.add_subplot(gs[1, 0])
    axC.scatter(y_theory, y_meas, s=6, color=fs.ACCENT, alpha=0.5, linewidths=0)
    lo, hi = 0, max(y_theory.max(), y_meas.max()) * 1.05
    axC.plot([lo, hi], [lo, hi], color=fs.C_I, lw=1.2, zorder=0)
    axC.set(xlabel="predicted rate [Hz]", ylabel="measured rate [Hz]",
            title="C: rate, predicted vs simulated", xlim=(lo, hi), ylim=(lo, hi))
    fs.despine(axC)

    # D: measured vs predicted correlation ---------------------------------------
    axD = fig.add_subplot(gs[1, 1])
    x = C_theory[off]; y = C_meas[off]
    sub = rng.choice(len(x), size=min(20000, len(x)), replace=False)
    axD.scatter(x[sub], y[sub], s=4, color=fs.ACCENT2, alpha=0.35, linewidths=0)
    lo2, hi2 = np.quantile(np.r_[x, y], [0.001, 0.999])
    axD.plot([lo2, hi2], [lo2, hi2], color=fs.C_I, lw=1.2, zorder=0)
    r = np.corrcoef(x, y)[0, 1]
    axD.set(xlabel="predicted correlation", ylabel=f"measured correlation ({bin_ms:.0f}ms bins)",
            title=f"D: correlation, predicted vs simulated (r={r:.2f})",
            xlim=(lo2, hi2), ylim=(lo2, hi2))
    fs.despine(axD)

    fig.suptitle("Hawkes ground truth reproduces its own closed-form theory "
                  "(cf. Pernice et al. 2011, Fig. 2)", fontsize=13, color=fs.INK)
    fs.save(fig, args.out)
    print(f"rate: theory mean={y_theory.mean():.2f}Hz  sim mean={y_meas.mean():.2f}Hz  "
          f"corr(theory,sim)={np.corrcoef(y_theory, y_meas)[0,1]:.3f}")
    print(f"correlation entries: corr(theory,sim)={r:.3f}")


if __name__ == "__main__":
    main()
