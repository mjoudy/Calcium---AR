"""
R.1 (method-demonstration panel) — PLOT stage, runs LOCALLY.

Reads the small r1_data.npz written on the cluster and renders the 2x3 panel:

    [ 3x3 confusion ]   [ ROC (+AUC) ]   [ PR (+AP) ]
    [ pdf by class  ]   [ cdf by class]   [ metrics box ]

Everything cosmetic (titles, labels, colours, size) is here — edit and re-run,
no cluster needed. Saves both a vector PDF (journal) and a PNG preview.

Usage:
  python scripts/fig_r1_plot.py --data results/fig_data/r1_data.npz \
      --out figures/fig_R1 --title "Method demonstration (canonical Brunel, N=12500)"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

CLASS_NAMES = ("E", "none", "I")


def draw_confusion(ax, cm_norm):
    ax.imshow(cm_norm, cmap=fs.BLUE_SEQ, vmin=0, vmax=1, aspect="equal")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.55 else fs.INK, fontsize=9)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("confusion (row-normalised)")
    fs.despine(ax, keep=())


def draw_roc(ax, z):
    ax.plot(z["grid"], z["tpr"], color=fs.ACCENT, lw=2)
    ax.plot([0, 1], [0, 1], ls=":", color=fs.MUTED, lw=1)
    ax.plot(z["op_fpr"], z["op_recall"], "o", ms=8, color=fs.ACCENT2,
            mec="white", mew=1.2, zorder=5)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="false positive rate",
           ylabel="true positive rate", title=f"ROC   AUC = {float(z['roc_auc']):.3f}")
    ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)


def draw_pr(ax, z):
    ax.plot(z["grid"], z["prec"], color=fs.ACCENT, lw=2)
    ax.plot(z["op_recall"], z["op_prec"], "o", ms=8, color=fs.ACCENT2,
            mec="white", mew=1.2, zorder=5)
    ax.axhline(float(z["true_density"]), ls=":", color=fs.MUTED, lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="recall", ylabel="precision",
           title=f"precision-recall   AP = {float(z['pr_ap']):.3f}")
    ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)


def draw_pdf(ax, z):
    centers = 0.5 * (z["edges"][:-1] + z["edges"][1:])
    for name, key in [("I", "pdf_I"), ("none", "pdf_none"), ("E", "pdf_E")]:
        ax.plot(centers, z[key], color=fs.CLASS_COLOR[name], lw=1.8, label=name)
    tau = float(z["tau"])
    ax.axvline(tau, color=fs.MUTED, ls="--", lw=0.9)
    ax.axvline(-tau, color=fs.MUTED, ls="--", lw=0.9)
    ax.set_yscale("log")
    ax.set(xlabel="inferred weight", ylabel="density (log)",
           title="weight distribution by true class")
    ax.legend(title="true class", loc="upper right")
    ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)


def draw_cdf(ax, z):
    for name, key in [("I", "cdf_I"), ("none", "cdf_none"), ("E", "cdf_E")]:
        ax.plot(z[key], z["cdf_y"], color=fs.CLASS_COLOR[name], lw=1.8, label=name)
    tau = float(z["tau"])
    ax.axvline(tau, color=fs.MUTED, ls="--", lw=0.9)
    ax.axvline(-tau, color=fs.MUTED, ls="--", lw=0.9)
    ax.set(ylim=(0, 1), xlabel="inferred weight", ylabel="CDF",
           title="cumulative distribution")
    ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)


def draw_metrics(ax, z):
    ax.axis("off")
    g = lambda k: float(z[k])
    lines = [
        ("Ranking (threshold-free)", ""),
        ("  correlation", f"{g('corr'):.3f}"),
        ("  ROC-AUC", f"{g('roc_auc'):.3f}"),
        ("  PR-AP", f"{g('pr_ap'):.3f}"),
        ("  AUC  E-vs-rest", f"{g('auc_E'):.3f}"),
        ("  AUC  I-vs-rest", f"{g('auc_I'):.3f}"),
        ("  neuron-type acc.", f"{g('type_acc'):.3f}"),
        (f"At {int(g('density')*100)}% density", ""),
        ("  macro precision / recall", f"{g('macro_p'):.2f} / {g('macro_r'):.2f}"),
        ("  excitatory  P / R", f"{g('E_p'):.2f} / {g('E_r'):.2f}"),
        ("  inhibitory  P / R", f"{g('I_p'):.2f} / {g('I_r'):.2f}"),
    ]
    y = 0.98
    for name, val in lines:
        head = val == ""
        ax.text(0.02, y, name, transform=ax.transAxes, va="top", ha="left",
                fontsize=10 if head else 9.5,
                fontweight="bold" if head else "normal",
                color=fs.INK if head else fs.MUTED)
        if val:
            ax.text(0.98, y, val, transform=ax.transAxes, va="top", ha="right",
                    fontsize=9.5, color=fs.INK, family="monospace")
        y -= 0.088
    ax.set_title("metrics")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/fig_data/r1_data.npz")
    ap.add_argument("--out", default="figures/fig_R1")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    fs.apply_style()
    z = np.load(args.data, allow_pickle=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.09,
                        hspace=0.42, wspace=0.32)
    draw_confusion(axes[0, 0], z["cm_norm"])
    draw_roc(axes[0, 1], z)
    draw_pr(axes[0, 2], z)
    draw_pdf(axes[1, 0], z)
    draw_cdf(axes[1, 1], z)
    draw_metrics(axes[1, 2], z)

    title = args.title or (f"Method demonstration  —  {str(z['label'])}, "
                           f"{str(z['method']).upper()}  (N={int(z['N'])})")
    fig.suptitle(title, fontsize=13, color=fs.INK, x=0.06, ha="left", y=0.975)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
