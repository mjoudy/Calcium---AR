"""
Effect of preprocessing — PLOT (local), consistent style.

    left     : metric comparison, raw calcium vs deconvolved feed (corr, AUC_E,
               excitatory recall).
    right x3 : per-class |inferred weight| PDFs (raw vs feed, one panel per
               class) — how deconvolution shifts each class's weight
               distribution relative to the unconnected baseline.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="figures/fig_preproc")
    args = ap.parse_args()

    fs.apply_style()
    z = np.load(args.data, allow_pickle=False)
    N = int(z["N"])

    fig, ax = plt.subplots(1, 4, figsize=(19, 5.2),
                           gridspec_kw={"width_ratios": [1.3, 1, 1, 1]})
    fig.subplots_adjust(left=0.045, right=0.985, top=0.86, bottom=0.14, wspace=0.32)

    # left: metrics raw vs feed
    keys = [("corr", "correlation"), ("auc_E", "AUC excitatory"),
            ("E_rec", "excitatory recall")]
    x = np.arange(len(keys)); w = 0.36
    raw = [float(z[f"raw_{k}"]) for k, _ in keys]
    feed = [float(z[f"feed_{k}"]) for k, _ in keys]
    ax[0].bar(x - w/2, raw, w, color=fs.C_NONE, label="raw calcium")
    ax[0].bar(x + w/2, feed, w, color=fs.C_E, label="deconvolved")
    for xi, (r, f) in enumerate(zip(raw, feed)):
        ax[0].text(xi - w/2, r + 0.01, f"{r:.2f}", ha="center", va="bottom", fontsize=8.5)
        ax[0].text(xi + w/2, f + 0.01, f"{f:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax[0].set_xticks(x); ax[0].set_xticklabels([lab for _, lab in keys], fontsize=9.5)
    ax[0].set(ylabel="score", ylim=(0, 1.05))
    ax[0].grid(True, axis="y", color=fs.GRID, lw=0.6); ax[0].set_axisbelow(True)
    fs.despine(ax[0]); ax[0].legend(fontsize=9, loc="upper right")
    ax[0].set_title("recovery: raw calcium vs deconvolved")

    # right 3 panels: per-class |weight| PDFs, raw (light) vs feed (dark) —
    # small multiples so the shift from deconvolution reads directly as a
    # peak moving right / narrowing, one class at a time instead of 6
    # overlapping CDF lines.
    for k, (cls, lab, col) in enumerate([("E", "excitatory", fs.C_E),
                                         ("I", "inhibitory", fs.C_I),
                                         ("none", "unconnected", fs.C_NONE)]):
        axp = ax[k + 1]
        feed, raw = z[f"feed_{cls}"], z[f"raw_{cls}"]
        vmax = np.percentile(np.concatenate([feed, raw]), 99)
        bins = np.linspace(0, vmax, 36)
        axp.hist(raw, bins=bins, density=True, color=col, alpha=0.35,
                 label="raw")
        axp.hist(feed, bins=bins, density=True, color=col, alpha=0.85,
                 histtype="step", lw=2.2, label="deconvolved")
        axp.set_title(lab, fontsize=10.5)
        axp.set_xlabel("|inferred weight|", fontsize=8.5)
        if k == 0:
            axp.set_ylabel("density")
        axp.grid(True, color=fs.GRID, lw=0.6); axp.set_axisbelow(True)
        fs.despine(axp)
        axp.legend(fontsize=8)

    fig.suptitle(f"Effect of preprocessing at N={N}: deconvolution and edge "
                 "recovery",
                 fontsize=13, color=fs.INK, x=0.07, ha="left", y=0.965)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
