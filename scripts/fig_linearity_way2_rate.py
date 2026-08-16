"""
Same three-way (LIF/PIF/OU) linearity comparison as fig_linearity_way2.py, but
using a THRESHOLDED/count-based measure instead of false-positive |weight|
magnitude: for every genuinely UNCONNECTED pair, binned by # common
presynaptic drivers, what FRACTION gets wrongly predicted "connected" (top-
density |weight|)? I.e. false-positive RATE vs shared-driver exposure, not
false-positive STRENGTH.

This uses the full population of non-edges (not just the ones that already
crossed threshold), and -- unlike raw |weight| -- a fraction (0-100%) is
already on a common, physically comparable scale across all three ground
truths, so no "%% change from lowest bin" normalization trick is needed here.

Usage:
  python scripts/fig_linearity_way2_rate.py --out figures/fig_linearity_way2_rate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

DEFAULT_SOURCES = [
    ("LIF", "~/calcium_results/best_moments/n1250r4", "#2a78d6"),
    ("PIF pilot (tau_m x10)",
     str(Path(__file__).resolve().parent.parent / "results/wrapup_n1250pif_T100k/seed1"),
     "#e8a33d"),
    ("PIF pilot (tau_m x100)", "~/calcium_results/wrapup_n1250pif100_T100k/seed1",
     "#b8860b"),
    ("Hawkes", "~/calcium_results/hawkes_moments/n1250_calcium_recal",
     "#2ca02c"),
]


def binned_rate(count, pred, min_n):
    """fraction of NON-EDGES predicted connected, per integer driver count."""
    vals = np.arange(count.min(), count.max() + 1)
    xs, rate, ns = [], [], []
    for v in vals:
        m = count == v
        if m.sum() >= min_n:
            xs.append(v); rate.append(100.0 * pred[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(rate)


def one_curve(data_dir, obs_frac, density, min_n, seed):
    d = Path(data_dir).expanduser()
    Cxx = np.load(d / "Cxx.npy"); Cyx = np.load(d / "Cyx.npy")
    adj = np.load(d / "adj_true.npy").astype(np.float64)
    N = Cxx.shape[0]; np.fill_diagonal(adj, 0.0)
    rng = np.random.default_rng(seed)

    types = np.sign(adj.sum(1)); types[types == 0] = 1
    E = np.flatnonzero(types > 0); I = np.flatnonzero(types < 0)
    rng.shuffle(E); rng.shuffle(I)
    nE, nI = int(obs_frac * len(E)), int(obs_frac * len(I))
    S = np.sort(np.concatenate([E[:nE], I[:nI]]))
    nS = len(S)

    ix = np.ix_(S, S)
    A = Cyx[ix] @ np.linalg.inv(Cxx[ix] + 1e-9 * np.eye(nS))
    np.fill_diagonal(A, 0.0)
    aa = np.abs(A)
    off = ~np.eye(nS, dtype=bool)
    tau = float(np.quantile(aa[off], 1.0 - density))
    pred = (aa > tau) & off
    truth = (adj[np.ix_(S, S)].T != 0) & off
    nonedge = off & ~truth                              # ALL genuine non-edges

    ia, ib = np.nonzero(nonedge)
    ti, tj = S[ia], S[ib]
    is_pred = pred[ia, ib]

    Bc = (adj != 0)
    both = Bc[:, ti] & Bc[:, tj]
    driver_count = both.sum(0)                          # total, ALL N sources

    return binned_rate(driver_count, is_pred, min_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--min-n", type=int, default=100, help="min non-edges per bin")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figures/fig_linearity_way2_rate")
    args = ap.parse_args()

    fs.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.87, bottom=0.13, wspace=0.22)

    for ax, obs_frac, title in zip(axes, [0.5, 1.0],
                                    ["50% observed", "100% observed"]):
        for label, path, color in DEFAULT_SOURCES:
            if not (Path(path).expanduser() / "Cxx.npy").exists():
                print(f"[skip] {label} @ {title}: no data yet at {path}")
                continue
            xs, rate = one_curve(path, obs_frac, args.density, args.min_n, args.seed)
            if len(xs) == 0:
                print(f"[skip] {label} @ {title}: no bins with >= {args.min_n} non-edges")
                continue
            ax.plot(xs, rate, "o-", color=color, lw=2, ms=5, label=label)
            print(f"{label:24s} @ {title:14s}: n_bins={len(xs)}  "
                  f"rate {rate[0]:.2f}% -> {rate[-1]:.2f}%  (x{rate[-1]/max(rate[0],1e-9):.2f})")
        ax.axhline(args.density * 100, color="#888", lw=1.0, ls="--", zorder=0)
        ax.set(xlabel="# common presynaptic drivers", title=title,
               ylabel="false-positive RATE\n% of non-edges predicted connected")
        ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)

    axes[0].legend(fontsize=9.5, loc="upper left")
    axes[0].text(0.98, 0.02, f"dashed: {args.density*100:.0f}% = chance/base rate",
                 transform=axes[0].transAxes, fontsize=8, color="#888",
                 ha="right", va="bottom")
    fig.suptitle("False-positive RATE vs shared-driver exposure, N=1250",
                  fontsize=13, color=fs.INK)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
