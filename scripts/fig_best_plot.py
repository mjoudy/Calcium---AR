"""
Best-results showcase — PLOT (local).

2 rows (ground truth / inferred) x 2 cols (full block-averaged view / sub-block).
Diverging colour: red = excitatory (+), blue = inhibitory (-). Each column shares
a symmetric colour scale set by a high percentile so the sparse structure shows.

Usage:
  python scripts/fig_best_plot.py --data ~/calcium_results/best/best_n12500lr.npz \
      --out figures/fig_best_n12500lr --title "N=12500, 14 Hz AI, OLS"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs


def show(fig, ax, M, pct, title, ylabel=None, sub_split=None):
    vmax = np.percentile(np.abs(M), pct) or 1e-6   # each panel scaled to itself
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal",
                   interpolation="nearest")
    if sub_split is not None:                    # mark exc|inh boundary
        for xy in ("axhline", "axvline"):
            getattr(ax, xy)(sub_split - 0.5, color=fs.INK, lw=0.8, ls=":")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="figures/fig_best")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    fs.apply_style()
    z = np.load(args.data, allow_pickle=False)
    N = int(z["N"]); split = int(z["n_exc_sub"])

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 10))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.9, bottom=0.04,
                        hspace=0.12, wspace=0.16)

    show(fig, ax[0][0], z["full_gt"], 99.5, "full network  (block-averaged)",
         ylabel="ground truth")
    show(fig, ax[1][0], z["full_est"], 99.5, "", ylabel="inferred (OLS)")
    show(fig, ax[0][1], z["sub_gt"], 99,
         f"sub-block  ({z['sub_size']} neurons, exc + inh)", sub_split=split)
    show(fig, ax[1][1], z["sub_est"], 99, "", sub_split=split)

    fig.text(0.5, 0.945, "red = excitatory (+)   blue = inhibitory (-)   "
             "dotted = exc|inh boundary   (each panel scaled to its own range)",
             ha="center", fontsize=9, color=fs.INK)

    title = args.title or f"Ground truth vs inferred connectivity (N={N})"
    fig.suptitle(title, fontsize=13.5, color=fs.INK, y=0.99)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
