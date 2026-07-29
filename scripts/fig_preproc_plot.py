"""
Effect of preprocessing — PLOT (local), consistent style.

    left  : metric comparison, raw calcium vs deconvolved feed (corr, AUC_E,
            excitatory recall).
    right : class-conditional CDFs of |inferred weight| for the deconvolved feed —
            how well excitatory / inhibitory separate from unconnected (CDF, not
            log-hist, per the professor's preference).

Usage:
  python scripts/fig_preproc_plot.py --data ~/calcium_results/preproc/preproc_n1250r4.npz \
      --out figures/fig_preproc_n1250
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs


def cdf(v):
    return np.sort(v), np.linspace(0, 1, len(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="figures/fig_preproc")
    args = ap.parse_args()

    fs.apply_style()
    z = np.load(args.data, allow_pickle=False)
    N = int(z["N"])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.14, wspace=0.24)

    # left: metrics raw vs feed
    keys = [("corr", "correlation"), ("auc_E", "AUC excitatory"),
            ("E_rec", "excitatory recall")]
    x = np.arange(len(keys)); w = 0.36
    raw = [float(z[f"raw_{k}"]) for k, _ in keys]
    feed = [float(z[f"feed_{k}"]) for k, _ in keys]
    ax[0].bar(x - w/2, raw, w, color=fs.C_NONE, label="raw calcium")
    ax[0].bar(x + w/2, feed, w, color=fs.C_E, label="deconvolved feed")
    for xi, (r, f) in enumerate(zip(raw, feed)):
        ax[0].text(xi - w/2, r + 0.01, f"{r:.2f}", ha="center", va="bottom", fontsize=8.5)
        ax[0].text(xi + w/2, f + 0.01, f"{f:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax[0].set_xticks(x); ax[0].set_xticklabels([lab for _, lab in keys], fontsize=9.5)
    ax[0].set(ylabel="score", ylim=(0, 1.05))
    ax[0].grid(True, axis="y", color=fs.GRID, lw=0.6); ax[0].set_axisbelow(True)
    fs.despine(ax[0]); ax[0].legend(fontsize=9, loc="upper right")
    ax[0].set_title("recovery: raw calcium vs deconvolved feed")

    # right: class-conditional |weight| CDFs for the feed
    for cls, lab, col in [("E", "excitatory edges", fs.C_E),
                          ("I", "inhibitory edges", fs.C_I),
                          ("none", "unconnected", fs.C_NONE)]:
        key = f"feed_{cls}"
        if key in z.files and len(z[key]):
            xv, yv = cdf(z[key])
            ax[1].plot(xv, yv, color=col, lw=2, label=lab)
    ax[1].set(xlabel="|inferred weight|  (normalised)", ylabel="cumulative fraction",
              ylim=(0, 1))
    ax[1].set_xlim(left=0)
    ax[1].grid(True, color=fs.GRID, lw=0.6); ax[1].set_axisbelow(True); fs.despine(ax[1])
    ax[1].legend(fontsize=9, loc="lower right")
    ax[1].set_title("weight separation by class (deconvolved feed)")

    fig.suptitle(f"Effect of preprocessing at N={N}: deconvolution and edge "
                 "recovery",
                 fontsize=13, color=fs.INK, x=0.07, ha="left", y=0.965)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
