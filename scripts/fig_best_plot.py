"""
Best-results showcase — PLOT (local).

2 rows (ground truth / inferred) x 2 cols (full block-averaged view / sub-block).
Diverging colour: red = excitatory (+), blue = inhibitory (-). Each column shares
a symmetric colour scale set by a high percentile so the sparse structure shows.

Also, if the compute stage wrote a `sub_err` array (categorical TN/TP-E/TP-I/FP/FN/
wrong-sign classification of the sub-block against a density-quantile threshold),
renders a companion 1x3 "error map" figure — ground truth, inferred (continuous),
and the classification with FP/FN in high-contrast colours against a muted
background, so the errors are the thing that pops out rather than the raw weights.

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
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

# --- categorical error-map palette: TN/TP muted, FP/FN loud & distinct ---------- #
ERR_LABELS = ["TN", "TP (E)", "TP (I)", "FP", "FN", "wrong sign"]
ERR_COLORS = ["#f2f1ec", "#a9c6ea", "#eeb3b0", "#e8890c", "#149e91", "#8b3fa8"]


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


def draw_errors(ax, code, sub_split, title):
    cmap = ListedColormap(ERR_COLORS)
    norm = BoundaryNorm(np.arange(len(ERR_COLORS) + 1) - 0.5, cmap.N)
    ax.imshow(code, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
    if sub_split is not None:
        for xy in ("axhline", "axvline"):
            getattr(ax, xy)(sub_split - 0.5, color=fs.INK, lw=0.8, ls=":")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)


def plot_error_map(z, out):
    if "sub_err" not in z.files:
        return
    split = int(z["n_exc_sub"])
    tau, density = float(z["tau"]), float(z["err_density"])

    fig, ax = plt.subplots(1, 3, figsize=(15, 5.6))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.84, bottom=0.14, wspace=0.12)

    show(fig, ax[0], z["sub_gt"], 99, "ground truth", sub_split=split)
    show(fig, ax[1], z["sub_est"], 99, "inferred (continuous)", sub_split=split)
    draw_errors(ax[2], z["sub_err"], split,
                f"errors  (density={density:.0%}, tau={tau:.2g})")

    handles = [mpatches.Patch(color=c, label=l) for c, l in zip(ERR_COLORS, ERR_LABELS)]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=9.5)

    fig.suptitle(f"Error map — sub-block ({z['sub_size']} neurons, exc + inh)",
                 fontsize=13, color=fs.INK, y=0.97)
    fs.save(fig, out)


def show_entries_bar(ax, vals, title):
    """Bar chart of one neuron's full inferred outgoing profile (all N-1
    targets, in target-index order), each bar colored by its own sign:
    positive = blue, negative = red."""
    colors = np.where(vals >= 0, fs.C_E, fs.C_I)
    ax.bar(np.arange(len(vals)), vals, color=colors, width=1.0)
    ax.axhline(0, color=fs.MUTED, lw=0.6)
    fs.despine(ax)
    ax.set_xticks([])
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("target neuron index", fontsize=8.5)


def plot_sample_entries(z, out):
    idx = z["sample_idx"]; is_exc = z["sample_is_exc"]; est = z["sample_est"]
    n = len(idx)
    if n == 0:
        return
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for k in range(min(n, 4)):
        kind = "exc" if is_exc[k] else "inh"
        show_entries_bar(ax.flat[k], est[k], f"neuron #{int(idx[k])} ({kind})")
    fig.text(0.5, 0.965, "blue = positive   red = negative", ha="center",
             fontsize=9, color=fs.INK)
    fig.suptitle("Sample neurons — full inferred outgoing profile",
                 fontsize=13, color=fs.INK, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fs.save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="figures/fig_best")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    fs.apply_style()
    z = np.load(args.data, allow_pickle=False)
    N = int(z["N"]); split = int(z["n_exc_sub"])
    b = int(z["block_factor"]) if "block_factor" in z.files else 1
    full_title = (f"full network  (block-averaged, {b}x{b})" if b > 2
                  else "full network")

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 10))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.9, bottom=0.04,
                        hspace=0.12, wspace=0.16)

    show(fig, ax[0][0], z["full_gt"], 99.5, full_title,
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

    if "sample_idx" in z.files:
        plot_sample_entries(z, f"{args.out}_entries")

    plot_error_map(z, f"{args.out}_errors")


if __name__ == "__main__":
    main()
