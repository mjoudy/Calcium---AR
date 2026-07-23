"""
R.4 (data length x network size) — the scaling law, plotted LOCALLY.

Reads the small metrics.csv files produced on the cluster (one per size x
recording length) and draws:

    left  column : measure vs recording length      -> bigger N needs more data
    right column : measure vs T/N (samples/neuron)  -> the curves COLLAPSE

rows = correlation and excitatory recall (the two measures with dynamic range;
ROC-AUC saturates and squashes the collapse against the ceiling).

All four sizes share one AI regime (13.9-14.8 Hz, CV 0.98-1.06, synchrony
0.007-0.010), so N is the only variable. OLS only.

Usage:
  python scripts/fig_r4_plot.py --root /home/mjoudy/calcium_results/hpc_metrics \
      --out figures/fig_R4
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

DT = 0.1  # ms per sample

# N -> directory prefix (the R.4 ladder; N=12500 reuses the low-rate A2 run)
LADDER = [(1250, "wrapup_n1250r4_T"), (2500, "wrapup_n2500r4_T"),
          (5000, "wrapup_n5000r4_T"), (12500, "wrapup_n12500r4_T")]
MEASURES = [("corr", "correlation"), ("E_rec", "excitatory recall")]


def load(root: Path, method="ols"):
    """-> {N: (T_ms[], {measure: values[]})}"""
    out = {}
    for N, prefix in LADDER:
        pts = []
        for d in sorted(root.glob(f"{prefix}*k")):
            f = d / "metrics.csv"
            if not f.exists():
                continue
            T_ms = float(d.name.rsplit("_T", 1)[1].rstrip("k")) * 1000.0
            with open(f) as fh:
                for row in csv.DictReader(fh):
                    if row["method"] == method:
                        pts.append((T_ms, {k: float(row[k]) for k, _ in MEASURES}))
                        break
        if pts:
            pts.sort()
            T = np.array([p[0] for p in pts])
            vals = {k: np.array([p[1][k] for p in pts]) for k, _ in MEASURES}
            out[N] = (T, vals)
        else:
            print(f"[warn] no metrics for N={N} ({prefix}*)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/mjoudy/calcium_results/hpc_metrics")
    ap.add_argument("--out", default="figures/fig_R4")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    fs.apply_style()
    data = load(Path(args.root))
    if not data:
        raise SystemExit(f"no metrics.csv found under {args.root}")
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(data)))

    fig, axes = plt.subplots(len(MEASURES), 2, figsize=(12.5, 8), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.89, bottom=0.09,
                        hspace=0.30, wspace=0.24)

    for r, (key, label) in enumerate(MEASURES):
        for (N, (T, vals)), c in zip(sorted(data.items()), colors):
            axes[r][0].plot(T, vals[key], "o-", color=c, lw=1.9, ms=6,
                            label=f"N = {N}")
            axes[r][1].plot((T / DT) / N, vals[key], "o-", color=c, lw=1.9, ms=6)
        for col, xlab in enumerate(["recording length (ms)",
                                    "samples per neuron   T / N"]):
            ax = axes[r][col]
            ax.set_xscale("log")
            ax.set(xlabel=xlab, ylabel=label, ylim=(0, 1.02))
            ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True)
            fs.despine(ax)
    axes[0][0].legend(fontsize=9, loc="lower right")
    axes[0][0].set_title("bigger networks need longer recordings")
    axes[0][1].set_title("...but collapse onto samples per neuron")

    title = args.title or ("Scaling: recovery is set by data per neuron "
                           "(matched AI regime, ~14 Hz, OLS)")
    fig.suptitle(title, fontsize=13, color=fs.INK, x=0.08, ha="left", y=0.965)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
