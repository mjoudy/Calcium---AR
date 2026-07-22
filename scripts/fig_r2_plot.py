"""
R.2 (calcium-observation section) — PLOT stage, runs LOCALLY.

Reads r2_data.npz and renders a 2x2 panel (rows = ROC-AUC, correlation):

  column A  vs dt/tau : the collapse test
      deconv (tau-swept) + deconv (rate-swept)   -> should be flat & overlap
      raw    (tau-swept) + raw    (rate-swept)   -> should fall with dt/tau
      dashed line = binned-spike ceiling (best)
  column B  vs spike bin size : the ceiling curve (DAAD-style ~5 ms optimum)

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

# line styles for the four dt/tau curves
STYLE = {
    "deconv_tau":  dict(color="#2a78d6", marker="o", label="deconvolved (dye τ)"),
    "deconv_rate": dict(color="#2a78d6", marker="s", ls="--", label="deconvolved (camera dt)"),
    "raw_tau":     dict(color="#e34948", marker="o", label="raw calcium (dye τ)"),
    "raw_rate":    dict(color="#e34948", marker="s", ls="--", label="raw calcium (camera dt)"),
}


def dt_tau_panel(ax, z, measure):
    for key, st in STYLE.items():
        x, y = z[f"{key}_x"], z[f"{key}_{measure}"]
        ax.plot(x, y, lw=1.8, ms=6, **st)
    ceiling = np.nanmax(z[f"spikes_{measure}"])       # best binned-spikes score
    ax.axhline(ceiling, color=fs.MUTED, ls=":", lw=1.2)
    ax.text(0.02, ceiling, "best binned spikes", color=fs.MUTED, va="bottom",
            ha="left", fontsize=8.5, transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set(xlabel="dt / τ   (calcium blur)",
           ylabel="ROC-AUC" if measure == "auc" else "correlation",
           ylim=(0, 1.02))
    ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)


def spike_panel(ax, z, measure):
    ax.plot(z["spikes_x"], z[f"spikes_{measure}"], "o-", color=fs.INK, lw=1.8, ms=6)
    ax.set_xscale("log")
    ax.set(xlabel="spike bin size (ms)",
           ylabel="ROC-AUC" if measure == "auc" else "correlation", ylim=(0, 1.02))
    ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/fig_data/r2_data.npz")
    ap.add_argument("--out", default="figures/fig_R2")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    fs.apply_style()
    z = np.load(args.data, allow_pickle=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8),
                             gridspec_kw=dict(width_ratios=[1.5, 1.0]))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.09,
                        hspace=0.32, wspace=0.26)
    dt_tau_panel(axes[0, 0], z, "auc");  axes[0, 0].set_title("blur (dt/τ): does only the ratio matter?")
    dt_tau_panel(axes[1, 0], z, "corr")
    spike_panel(axes[0, 1], z, "auc");   axes[0, 1].set_title("binned spikes (reference)")
    spike_panel(axes[1, 1], z, "corr")
    axes[0, 0].legend(loc="lower left", fontsize=8.5)

    title = args.title or (f"Calcium observation  —  {str(z['net'])}, "
                           f"N={int(z['N'])}, T={int(z['T_ms'])//1000}k ms")
    fig.suptitle(title, fontsize=13, color=fs.INK, x=0.07, ha="left", y=0.975)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
