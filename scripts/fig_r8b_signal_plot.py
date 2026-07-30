"""
Timing signature: spikes vs calcium — PLOT (local).

Reads two R.8b npz for the same network — one built on raw SPIKES, one on the
deconvolved CALCIUM feed — and shows the class-averaged lag profiles side by side.

The point: the timing idea WORKS on spikes (a true edge peaks at the synaptic
delay; a shared-input false positive peaks at ~0, instantaneous — the curves
separate) but is DESTROYED by the calcium blur (both peak at the same lag — the
curves merge). So the idea is sound in principle; calcium observation is what
defeats it.

Usage:
  python scripts/fig_r8b_signal_plot.py \
      --spikes ~/calcium_results/r8b/r8b_n1250_r4_spikes.npz \
      --calcium ~/calcium_results/r8b/r8b_n1250_r4_feed.npz \
      --out figures/fig_R8b_signal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs


def panel(ax, z, title):
    lags = z["lags_ms"]; delay = float(z["delay_ms"])
    for key, lab, col in [("prof_Etrue", "true excitatory edges", fs.C_E),
                          ("prof_Efalse", "excitatory false positives", "#c0392b")]:
        if key in z.files:
            v = z[key]
            ax.plot(lags, v / np.nanmax(v), "o-", color=col, lw=2, ms=5, label=lab)
    ax.axvline(delay, color=fs.INK, ls=":", lw=1.3)
    ax.text(delay, 1.03, f" delay {delay:g}ms", fontsize=8.5, color=fs.INK)
    ax.set(xlabel="lag (ms)", ylabel="mean |A(lag)|  (normalised)", ylim=(0, 1.12))
    ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spikes", required=True)
    ap.add_argument("--calcium", required=True)
    ap.add_argument("--out", default="figures/fig_R8b_signal")
    args = ap.parse_args()

    fs.apply_style()
    zs = np.load(args.spikes, allow_pickle=False)
    zc = np.load(args.calcium, allow_pickle=False)
    N = int(zs["N"])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.85, bottom=0.14, wspace=0.22)
    panel(ax[0], zs, "on raw SPIKES: idea works\n(true peaks at delay, fake at ~0)")
    panel(ax[1], zc, "through CALCIUM: idea fails\n(both peak at the same lag)")

    fig.suptitle(f"Timing separates true edges from shared-input fakes on spikes, "
                 f"but the calcium blur erases it (N={N})",
                 fontsize=13, color=fs.INK, x=0.07, ha="left", y=0.965)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
