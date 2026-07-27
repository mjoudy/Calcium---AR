"""
R.5 confusion matrices — PLOT stage, local.

One figure per network size: a grid of 3x3 confusion matrices, rows = recording
length (T/N), columns = method. Each cell is row-normalised (so the diagonal is
per-class recall) and annotated with the raw edge counts underneath.

    rows of each 3x3 = TRUE class  (E, none, I)
    cols of each 3x3 = PREDICTED   (E, none, I)

Usage:
  python scripts/fig_r5_confusion_plot.py \
      --data ~/calcium_results/r5/r5conf_n1250r4.npz --out figures/fig_R5_conf_n1250
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

MLABEL = {"ols": "OLS", "lasso": "Lasso", "lassodale": "Lasso+Dale",
          "en": "EN", "endale": "EN+Dale", "ridge": "Ridge"}


def draw(ax, C, clabel, title):
    row = C / np.maximum(C.sum(1, keepdims=True), 1)   # recall-normalised
    im = ax.imshow(row, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    for ti in range(3):
        for pi in range(3):
            frac = row[ti, pi]
            ax.text(pi, ti - 0.12, f"{frac*100:.0f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if frac > 0.55 else fs.INK)
            ax.text(pi, ti + 0.24, f"{C[ti, pi]:,}", ha="center", va="center",
                    fontsize=6.5, color="white" if frac > 0.55 else "#666")
    ax.set_xticks(range(3)); ax.set_xticklabels(clabel)
    ax.set_yticks(range(3)); ax.set_yticklabels(clabel)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    if title:
        ax.set_title(title, fontsize=11)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fs.apply_style()
    z = np.load(args.data, allow_pickle=True)
    conf = z["conf"]                     # (n_tn, n_method, 3, 3)
    methods = [str(m) for m in z["methods"]]
    clabel = [str(c) for c in z["clabel"]]
    TN = z["TN"]; N = int(z["N"])
    n_tn, n_m = conf.shape[0], conf.shape[1]

    fig, axes = plt.subplots(n_tn, n_m, figsize=(3.4 * n_m, 3.4 * n_tn),
                             squeeze=False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.08,
                        hspace=0.45, wspace=0.45)
    for r in range(n_tn):
        for c in range(n_m):
            title = MLABEL.get(methods[c], methods[c]) if r == 0 else ""
            draw(axes[r][c], conf[r, c], clabel, title)
        axes[r][0].annotate(f"T/N = {TN[r]:.0f}", xy=(-0.55, 0.5),
                            xycoords="axes fraction", rotation=90, va="center",
                            ha="center", fontsize=12, color=fs.INK)

    fig.suptitle(f"N = {N}: confusion at the 10% operating point "
                 "(cell = recall %, small = edge count; best λ by E-recall)",
                 fontsize=13, color=fs.INK, x=0.10, ha="left", y=0.965)
    out = args.out or str(Path(args.data).with_suffix(""))
    fs.save(fig, out)


if __name__ == "__main__":
    main()
