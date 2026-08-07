"""
R.4b — fixed connection-PROBABILITY ladder (eps=0.1, C_E grows with N) vs
fixed IN-DEGREE ladder (C_E=100 fixed, eps shrinks with N), overlaid.

Tests whether the excitatory-recall/precision gap found in R.4 is a real,
scale-invariant-convention-independent effect, or an artifact of holding
connection probability (rather than in-degree) fixed across sizes. If the
"ci" (constant in-degree) lines collapse onto each other while the "r4"
(constant probability) lines stay separated, the R.4 finding was partly a
scaling-convention artifact -- real cortex is closer to fixed in-degree.

Solid = r4 ladder (fixed probability), dashed = ci ladder (fixed in-degree).
N=1250 is IDENTICAL in both ladders (same eps=0.1 either way) -- one line,
not two, at that size.

Usage:
  python scripts/fig_r4ci_plot.py --root ~/calcium_results/hpc_metrics \
      --out figures/fig_R4_CI
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs
from fig_r4_plot import load, MEASURES

LADDER_R4 = [(1250, "wrapup_n1250r4_T"), (2500, "wrapup_n2500r4_T"),
            (5000, "wrapup_n5000r4_T"), (12500, "wrapup_n12500r4_T")]
LADDER_CI = [(1250, "wrapup_n1250r4_T"),   # identical to r4 at this size
            (2500, "wrapup_n2500ci_T"), (5000, "wrapup_n5000ci_T"),
            (12500, "wrapup_n12500ci_T")]


def load_ladder(root, ladder):
    import fig_r4_plot as m
    m.LADDER = ladder          # load() reads the module-level LADDER
    return load(root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/mjoudy/calcium_results/hpc_metrics")
    ap.add_argument("--out", default="figures/fig_R4_CI")
    args = ap.parse_args()

    fs.apply_style()
    root = Path(args.root)
    data_r4 = load_ladder(root, LADDER_R4)
    data_ci = load_ladder(root, LADDER_CI)
    if not data_r4 or not data_ci:
        raise SystemExit("missing metrics for one or both ladders under " + str(root))

    Ns = sorted(data_r4)
    colors = dict(zip(Ns, plt.cm.viridis(np.linspace(0.15, 0.85, len(Ns)))))

    fig, axes = plt.subplots(len(MEASURES), 1, figsize=(7, 11), squeeze=False)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.91, bottom=0.06, hspace=0.30)

    for r, (key, label) in enumerate(MEASURES):
        ax = axes[r][0]
        for N in Ns:
            c = colors[N]
            T, vals = data_r4[N]
            mean, std, _ = vals[key]
            ax.errorbar(T, mean, yerr=std, fmt="o-", color=c, lw=1.9, ms=6,
                        capsize=3, elinewidth=1.2,
                        label=f"N={N} (fixed prob.)" if r == 0 else None)
            if N == 1250:
                continue   # identical line, don't double-plot
            if N not in data_ci:
                continue   # ci data for this size not run yet
            T2, vals2 = data_ci[N]
            mean2, std2, _ = vals2[key]
            ax.errorbar(T2, mean2, yerr=std2, fmt="s--", color=c, lw=1.9, ms=6,
                        capsize=3, elinewidth=1.2, alpha=0.75,
                        label=f"N={N} (fixed in-deg.)" if r == 0 else None)
        ax.set_xscale("log")
        ax.set(xlabel="recording length (ms)", ylabel=label, ylim=(0, 1.02))
        ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True)
        fs.despine(ax)
    axes[0][0].legend(fontsize=8, loc="lower right", ncol=2)

    fig.suptitle("Does the N-scaling gap survive fixed in-degree scaling?  "
                 "(solid=fixed probability, dashed=fixed in-degree)",
                 fontsize=12.5, color=fs.INK, x=0.07, ha="left", y=0.965)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
