"""
Way 2: does false-positive strength still grow with shared-driver exposure as
the generative process gets more linear?

Same procedure as fig_way2.py (observe a fraction S, OLS-solve, threshold to
top-density |weight|, find false positives, bin against # common presynaptic
drivers) run identically on ground truths of increasing linearity at N=1250:
  - LIF (real spiking + calcium)          best_moments/n1250r4
  - PIF pilot (tau_m x10/x100, still spiking)  results/wrapup_n1250pif*_T100k
  - Hawkes (linear point process + calcium)    hawkes_moments/n1250_calcium_recal

Plotted as %% change in mean false-positive |weight| from the lowest driver-
count bin (not raw |weight| -- the ground truths have incomparable raw
units/scales, calcium signal vs abstract linear units) so the curves sit
on one shared, meaningful y-axis. Two panels: 50%% and 100%% observed.

Caveat: LIF/Hawkes share the exact same adjacency; PIF has its own
independently-drawn random topology at the same target density -- so this
compares the same STATISTICAL structure, not pair-for-pair identical graphs.

Usage:
  python scripts/fig_linearity_way2.py --out figures/fig_linearity_way2
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
    # tau_m x100 (2000ms), eta=80.0 from scripts/pif_tau_probe.py --stage x100
    # (2026-08-15) -- CV=6.14 at this stage, much more irregular again than
    # the x10 pilot's CV=1.94 (see the NETS["n1250_pif100"] comment in
    # wrapup_run.py).
    ("PIF pilot (tau_m x100)", "~/calcium_results/wrapup_n1250pif100_T100k/seed1",
     "#b8860b"),
    # Real point process (Pernice et al. 2011's own model class), run through
    # the SAME calcium+deconvolution pipeline as LIF/PIF (see
    # scripts/hawkes_ground_truth.py / hawkes_to_moments.py).
    ("Hawkes", "~/calcium_results/hawkes_moments/n1250_calcium_recal",
     "#2ca02c"),
]


def binned_pct(count, w, min_n):
    vals = np.arange(count.min(), count.max() + 1)
    xs, ys = [], []
    for v in vals:
        m = count == v
        if m.sum() >= min_n:
            xs.append(v); ys.append(w[m].mean())
    xs, ys = np.array(xs), np.array(ys)
    if len(ys) == 0:
        return xs, ys
    pct = (ys / ys[0] - 1.0) * 100.0
    return xs, pct


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
    fp = pred & ~truth
    fa, fb = np.nonzero(fp)
    ti, tj = S[fa], S[fb]
    w = aa[fa, fb]

    Bc = (adj != 0)
    both = Bc[:, ti] & Bc[:, tj]
    driver_count = both.sum(0)                          # total, ALL N sources

    return binned_pct(driver_count, w, min_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--min-n", type=int, default=100, help="min pairs per bin")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figures/fig_linearity_way2")
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
            xs, pct = one_curve(path, obs_frac, args.density, args.min_n, args.seed)
            if len(xs) == 0:
                print(f"[skip] {label} @ {title}: no bins with >= {args.min_n} pairs")
                continue
            ax.plot(xs, pct, "o-", color=color, lw=2, ms=5, label=label)
            print(f"{label:24s} @ {title:14s}: n_bins={len(xs)}  "
                  f"total rise={pct[-1]-pct[0]:.1f}pp")
        ax.set(xlabel="# common presynaptic drivers", title=title,
               ylabel="false-positive |weight|\n% change from lowest bin")
        ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)
        ax.axhline(0, color="#888", lw=0.8, zorder=0)

    axes[0].legend(fontsize=9.5, loc="upper left")
    fig.suptitle("Shared-input false positives vs ground-truth linearity, N=1250",
                  fontsize=13, color=fs.INK)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
