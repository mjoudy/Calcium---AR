"""
R.2 (calcium-observation section) — PLOT stage, runs LOCALLY.

Reads r2_data.npz and renders the PRIMARY figure as a 2x3 panel
(rows = ROC-AUC, correlation):

  column A  vs camera frame interval (ms), dye tau FIXED
      deconv_rate + raw_rate
  column B  vs dye tau (ms), camera rate FIXED
      deconv_tau  + raw_tau
  column C  vs spike bin size (ms) : the ceiling curve (DAAD-style ~5 ms optimum)

Camera and tau are swept ONE AT A TIME (see fig_r2_compute.py) so each column
answers one physical question on its own, instead of both being folded into a
shared dt/tau ratio axis.

A SECONDARY figure (--out-ratio) reproduces the old "does only the ratio
matter?" collapse view, for anyone who wants to check whether the two curves
line up when replotted against dt/tau after the fact.

All cosmetics live here. Saves PDF + PNG.

Usage:
  python scripts/fig_r2_plot.py --data results/fig_data/r2_data.npz --out figures/fig_R2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

STYLE_RATE = {
    "deconv_rate": dict(color="#2a78d6", marker="o", label="deconvolved"),
    "raw_rate":    dict(color="#e34948", marker="o", label="raw calcium"),
}
STYLE_TAU = {
    "deconv_tau": dict(color="#2a78d6", marker="o", label="deconvolved"),
    "raw_tau":    dict(color="#e34948", marker="o", label="raw calcium"),
}
STYLE_RATIO = {
    "deconv_tau":  dict(color="#2a78d6", marker="o", label="deconvolved (dye τ)"),
    "deconv_rate": dict(color="#2a78d6", marker="s", ls="--", label="deconvolved (camera dt)"),
    "raw_tau":     dict(color="#e34948", marker="o", label="raw calcium (dye τ)"),
    "raw_rate":    dict(color="#e34948", marker="s", ls="--", label="raw calcium (camera dt)"),
}


def _ceiling(ax, z, measure):
    ceiling = np.nanmax(z[f"spikes_{measure}"])       # best binned-spikes score
    ax.axhline(ceiling, color=fs.MUTED, ls=":", lw=1.2)
    ax.text(0.02, ceiling, "best binned spikes", color=fs.MUTED, va="bottom",
            ha="left", fontsize=8, transform=ax.get_yaxis_transform())


def rate_panel(ax, z, measure):
    """vs camera frame interval (ms), dye tau fixed."""
    for key, st in STYLE_RATE.items():
        x, y = z[f"{key}_x"], z[f"{key}_{measure}"]
        ax.plot(x, y, lw=1.8, ms=6, **st)
    _ceiling(ax, z, measure)
    ax.axvline(float(z["fixed_tau_ms"]), color=fs.MUTED, lw=1.0, ls="-.")
    ax.set_xscale("log")
    ax.set(xlabel="camera frame interval (ms)",
           ylabel="ROC-AUC" if measure == "auc" else "correlation", ylim=(0, 1.02))
    ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)


def tau_panel(ax, z, measure):
    """vs dye tau (ms), camera rate fixed."""
    for key, st in STYLE_TAU.items():
        x, y = z[f"{key}_x"], z[f"{key}_{measure}"]
        ax.plot(x, y, lw=1.8, ms=6, **st)
    _ceiling(ax, z, measure)
    cam = float(z["fixed_cam_ms"])
    ax.axvline(cam, color=fs.MUTED, lw=1.0, ls="-.")
    ax.text(cam, 0.02, " camera dt", color=fs.MUTED, fontsize=7.5, rotation=90,
            va="bottom", ha="left")
    ax.set_xscale("log")
    ax.set(xlabel="dye τ (ms)",
           ylabel="ROC-AUC" if measure == "auc" else "correlation", ylim=(0, 1.02))
    ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)


def spike_panel(ax, z, measure):
    ax.plot(z["spikes_x"], z[f"spikes_{measure}"], "o-", color=fs.INK, lw=1.8, ms=6)
    ax.set_xscale("log")
    ax.set(xlabel="spike bin size (ms)",
           ylabel="ROC-AUC" if measure == "auc" else "correlation", ylim=(0, 1.02))
    ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)


def ratio_panel(ax, z, measure):
    for key, st in STYLE_RATIO.items():
        x, y = z[f"{key}_ratio"], z[f"{key}_{measure}"]
        order = np.argsort(x)
        ax.plot(x[order], y[order], lw=1.8, ms=6, **st)
    _ceiling(ax, z, measure)
    ax.set_xscale("log")
    ax.set(xlabel="dt / τ   (calcium blur)",
           ylabel="ROC-AUC" if measure == "auc" else "correlation", ylim=(0, 1.02))
    ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)


def net_title(z):
    return f"{str(z['net'])}, N={int(z['N'])}, T={int(z['T_ms'])//1000}k ms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/fig_data/r2_data.npz")
    ap.add_argument("--out", default="figures/fig_R2")
    ap.add_argument("--out-ratio", default="figures/fig_R2_ratio",
                    help="secondary collapse-view figure (dt/tau shared axis)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    fs.apply_style()
    z = np.load(args.data, allow_pickle=False)

    # ---- primary: one variable at a time ---------------------------------- #
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8),
                             gridspec_kw=dict(width_ratios=[1.15, 1.15, 1.0]))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.88, bottom=0.09,
                        hspace=0.32, wspace=0.3)
    rate_panel(axes[0, 0], z, "auc");  axes[0, 0].set_title(
        f"camera rate  (dye τ fixed at {float(z['fixed_tau_ms']):.0f} ms)")
    rate_panel(axes[1, 0], z, "corr")
    tau_panel(axes[0, 1], z, "auc");   axes[0, 1].set_title(
        f"dye τ  (camera fixed at {float(z['fixed_cam_ms']):.0f} ms)")
    tau_panel(axes[1, 1], z, "corr")
    spike_panel(axes[0, 2], z, "auc"); axes[0, 2].set_title("binned spikes (reference)")
    spike_panel(axes[1, 2], z, "corr")
    axes[0, 0].legend(loc="lower left", fontsize=8.5)

    title = args.title or f"Calcium observation  —  {net_title(z)}"
    fig.suptitle(title, fontsize=13, color=fs.INK, x=0.055, ha="left", y=0.975)
    fs.save(fig, args.out)

    # ---- secondary: does the ratio alone explain it? ----------------------- #
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.6))
    fig2.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.14, wspace=0.28)
    ratio_panel(axes2[0], z, "auc");  axes2[0].set_title("blur (dt/τ): does only the ratio matter?")
    ratio_panel(axes2[1], z, "corr")
    axes2[0].legend(loc="lower left", fontsize=8)
    fig2.suptitle(f"Ratio collapse check  —  {net_title(z)}",
                 fontsize=12, color=fs.INK, x=0.08, ha="left", y=0.97)
    fs.save(fig2, args.out_ratio)


if __name__ == "__main__":
    main()
