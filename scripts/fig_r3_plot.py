"""
R.3 (lag & delay) — PLOT stage, local.

  A: AUC vs regression lag, one curve per true synaptic delay (peak marked)
  B: recovered optimal lag vs true delay (should track the delay; diagonal ref)

Usage:
  python scripts/fig_r3_plot.py --data results/fig_data/r3_data.npz --out figures/fig_R3
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
    ap.add_argument("--data", default="results/fig_data/r3_data.npz")
    ap.add_argument("--out", default="figures/fig_R3")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    fs.apply_style()
    z = np.load(args.data, allow_pickle=False)
    delays = z["delays"]; lags = z["lags_ms"]; opt = z["opt_lag"]
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(delays)))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.13, wspace=0.26)

    # A — lag sweep, one curve per true delay
    for D, c in zip(delays, cmap):
        auc = z[f"auc_D{D}"]
        a1.plot(lags, auc, "o-", color=c, lw=1.8, ms=5, label=f"D = {D:g} ms")
        a1.plot(lags[np.argmax(auc)], np.max(auc), "*", color=c, ms=13,
                mec="white", mew=0.8, zorder=5)
    a1.set_xscale("log")
    a1.set(xlabel="regression lag (ms)", ylabel="ROC-AUC",
           title="lag sweep (★ = peak per delay)")
    a1.grid(True, color=fs.GRID, lw=0.6); a1.set_axisbelow(True); fs.despine(a1)
    a1.legend(title="true delay", fontsize=8.5)

    # B — recovered optimal lag vs true delay
    lo = min(delays.min(), opt.min()); hi = max(delays.max(), opt.max())
    a2.plot([lo, hi], [lo, hi], ls=":", color=fs.MUTED, lw=1.2, label="identity")
    a2.plot(delays, opt, "o-", color=fs.ACCENT2, lw=2, ms=8)
    a2.set(xlabel="true synaptic delay (ms)", ylabel="recovered optimal lag (ms)",
           title="the method reads out the delay")
    a2.grid(True, color=fs.GRID, lw=0.6); a2.set_axisbelow(True); fs.despine(a2)
    a2.legend(fontsize=9)

    title = args.title or (f"Lag & synaptic-delay recovery  —  {str(z['net'])}")
    fig.suptitle(title, fontsize=13, color=fs.INK, x=0.07, ha="left", y=0.97)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
