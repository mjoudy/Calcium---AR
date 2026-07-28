"""
R.7 (hidden neurons / shared input) — PLOT stage, local.

    left  : recovery vs OBSERVED FRACTION, at matched samples per observed neuron
            (so any drop is hidden-input confounding, not less data). Excitatory
            recall/precision and, for contrast, inhibitory recall.
    right : MECHANISM — for observed pairs that are NOT truly connected, the mean
            inferred |weight| vs their number of common presynaptic neurons.
            A rising curve is shared input manufacturing false edges.

Usage:
  python scripts/fig_r7_plot.py --data ~/calcium_results/r7/r7_n12500r4.npz \
      --out figures/fig_R7
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
    ap.add_argument("--out", default="figures/fig_R7")
    ap.add_argument("--min-count", type=int, default=2000,
                    help="drop shared-input bins with fewer pairs than this")
    args = ap.parse_args()

    fs.apply_style()
    z = np.load(args.data, allow_pickle=False)
    N = int(z["N"]); frac = z["fractions"]
    order = np.argsort(frac)
    frac = frac[order]

    has_shared = "shared_bins" in z.files
    fig, axes = plt.subplots(1, 2 if has_shared else 1, figsize=(13 if has_shared else 7, 5.2),
                             squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.14, wspace=0.24)

    axL = axes[0][0]
    series = [("corr", "correlation", fs.INK, "-"),
              ("E_rec", "excitatory recall", fs.C_E, "-"),
              ("E_prec", "excitatory precision", fs.C_E, "--"),
              ("I_rec", "inhibitory recall", fs.C_I, "-")]
    for key, lab, col, ls in series:
        axL.plot(frac * 100, z[key][order], "o", ls=ls, color=col, lw=2, ms=6, label=lab)
    axL.set(xlabel="observed fraction of the network (%)", ylabel="score",
            ylim=(0, 1.02))
    axL.invert_xaxis()      # more hidden -> to the right
    axL.grid(True, color=fs.GRID, lw=0.6); axL.set_axisbelow(True); fs.despine(axL)
    axL.legend(fontsize=9, loc="lower left")
    axL.set_title("recovery vs observed fraction\n(matched samples per observed neuron)")

    if has_shared:
        axR = axes[0][1]
        b = z["shared_bins"]; w = z["shared_meanw"]; c = z["shared_count"]
        keep = c >= args.min_count
        axR.plot(b[keep], w[keep], "o-", color="#c0392b", lw=2, ms=5)
        axR.set(xlabel="number of common presynaptic neurons (shared input)",
                ylabel="mean inferred |weight| of NON-connected pairs")
        axR.grid(True, color=fs.GRID, lw=0.6); axR.set_axisbelow(True); fs.despine(axR)
        sf = float(z["shared_frac"]) * 100 if "shared_frac" in z.files else None
        axR.set_title("shared input manufactures false edges" +
                      (f"\n(observed fraction {sf:.0f}%)" if sf else ""))

    fig.suptitle(f"Hidden neurons and shared input at N={N} (OLS): "
                 "unobserved common input is the confounding source",
                 fontsize=13, color=fs.INK, x=0.07, ha="left", y=0.965)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
