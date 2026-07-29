"""
R.8 (shared input vs directionality) — PLOT stage, local.

    left  : DIAGNOSIS — CDF of the directional-asymmetry score for predicted
            edges that are TRUE (excitatory/inhibitory) vs FALSE POSITIVES. If the
            false-positive curve sits to the LEFT (lower asymmetry = more
            symmetric), shared input is the source and asymmetry separates it.
    right : EFFECT/FIX — excitatory precision and recall as symmetric edges are
            filtered out (asymmetry threshold). Precision up = fakes removed;
            recall down = the cost. The baseline (no filter) is the left endpoint.

Usage:
  python scripts/fig_r8_plot.py --data ~/calcium_results/r8/r8_wrapup_n12500r4_T5000k.npz \
      --out figures/fig_R8_n12500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs


def cdf_xy(v):
    v = np.sort(v)
    return v, np.linspace(0, 1, len(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="figures/fig_R8")
    args = ap.parse_args()

    fs.apply_style()
    z = np.load(args.data, allow_pickle=True)
    N = int(z["N"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.14, wspace=0.24)

    # left: asymmetry CDFs
    axL = axes[0]
    lines = [("asym_E_true", "true excitatory edges", fs.C_E, "-"),
             ("asym_E_false", "excitatory FALSE positives", fs.C_E, "--"),
             ("asym_I_true", "true inhibitory edges", fs.C_I, "-"),
             ("asym_I_false", "inhibitory FALSE positives", fs.C_I, "--")]
    for key, lab, col, ls in lines:
        if key in z.files and len(z[key]) > 0:
            x, y = cdf_xy(z[key])
            axL.plot(x, y, ls, color=col, lw=2, label=lab)
    axL.set(xlabel="directional asymmetry  |A[i,j]-A[j,i]| / (|A[i,j]|+|A[j,i]|)",
            ylabel="cumulative fraction", xlim=(0, 1), ylim=(0, 1))
    axL.grid(True, color=fs.GRID, lw=0.6); axL.set_axisbelow(True); fs.despine(axL)
    axL.legend(fontsize=8.5, loc="upper left")
    axL.set_title("diagnosis: are false positives more symmetric?")

    # right: precision/recall vs asym filter
    axR = axes[1]
    thr = z["thr"]
    axR.plot(thr, z["E_prec"], "-", color="#c0392b", lw=2.2, label="excitatory precision")
    axR.plot(thr, z["E_rec"], "-", color=fs.C_E, lw=2.2, label="excitatory recall")
    axR.axvline(0, color=fs.GRID, lw=1)
    axR.annotate("baseline\n(no filter)", xy=(0.02, z["E_prec"][0]),
                 fontsize=8.5, color=fs.INK, va="center")
    axR.set(xlabel="remove predicted edges with asymmetry below this",
            ylabel="score", xlim=(0, 1), ylim=(0, 1.02))
    axR.grid(True, color=fs.GRID, lw=0.6); axR.set_axisbelow(True); fs.despine(axR)
    axR.legend(fontsize=9, loc="upper right")
    axR.set_title("fix: filter symmetric edges, re-score")

    fig.suptitle(f"Shared input vs directionality at N={N} (single-lag OLS): "
                 "do symmetric edges flag the shared-input fakes?",
                 fontsize=13, color=fs.INK, x=0.07, ha="left", y=0.965)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
