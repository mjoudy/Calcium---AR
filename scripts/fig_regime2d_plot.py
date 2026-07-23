"""
2-D regime probe — PLOT stage, local.

Heatmaps of firing rate, ISI CV and synchrony over the (g, eta) grid, with the
two target contours overlaid (rate = target, CV = target). Where those two
contours CROSS is a configuration that is simultaneously realistic in rate and
textbook-AI in irregularity — the point of the whole probe.

Usage:
  python scripts/fig_regime2d_plot.py --data results/regime2d/regime2d_N1250_j1.npz \
      --out figures/fig_regime2d_N1250
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs


def panel(ax, M, g, eta, title, cmap, target=None, fmt="{:.2f}"):
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap,
                   extent=[-0.5, len(g) - 0.5, -0.5, len(eta) - 0.5])
    lo, hi = np.nanmin(M), np.nanmax(M)
    for i in range(len(eta)):
        for j in range(len(g)):
            if not np.isfinite(M[i, j]):
                continue
            r, gr, b, _ = im.cmap((M[i, j] - lo) / (hi - lo + 1e-9))
            lum = 0.299 * r + 0.587 * gr + 0.114 * b   # readable on any cmap
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                    fontsize=7.5, color="white" if lum < 0.5 else fs.INK)
    if target is not None and np.nanmin(M) < target < np.nanmax(M):
        ax.contour(np.arange(len(g)), np.arange(len(eta)), M, levels=[target],
                   colors="#e34948", linewidths=2.2)
    ax.set_xticks(range(len(g))); ax.set_xticklabels([f"{v:g}" for v in g])
    ax.set_yticks(range(len(eta))); ax.set_yticklabels([f"{v:g}" for v in eta])
    ax.set_xlabel("g  (inhibition dominance)"); ax.set_ylabel("eta  (drive)")
    ax.set_title(title)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fs.apply_style()
    z = np.load(args.data, allow_pickle=False)
    g, eta = z["g"], z["eta"]
    tr, tc = float(z["target_rate"]), float(z["target_cv"])

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.84, bottom=0.14, wspace=0.30)
    im0 = panel(axes[0], z["rate"], g, eta, f"firing rate (Hz)   red = {tr:g} Hz",
                "viridis", target=tr, fmt="{:.0f}")
    im1 = panel(axes[1], z["cv"], g, eta, f"ISI CV   red = {tc:g}", "magma",
                target=tc)
    im2 = panel(axes[2], z["sync"], g, eta, "synchrony (AI ~ 0)", "magma_r",
                fmt="{:.3f}")
    for ax, im in zip(axes, (im0, im1, im2)):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Regime map at N={int(z['N'])} (J={float(z['J']):.3f}, "
                 f"x{float(z['j_scale']):g})  —  where the two red contours cross, "
                 f"rate and CV targets are met together",
                 fontsize=12.5, color=fs.INK, x=0.05, ha="left", y=0.965)
    out = args.out or str(Path(args.data).with_suffix(""))
    fs.save(fig, out)


if __name__ == "__main__":
    main()
