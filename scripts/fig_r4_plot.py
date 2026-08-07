"""
R.4 (data length x network size) — the scaling law, plotted LOCALLY.

Reads the small metrics.csv files produced on the cluster (one per size x
recording length) and draws one column: measure vs recording length -> bigger
N needs more data. (The "vs T/N, curves collapse" column was dropped per
professor's feedback — it was somewhat trivially built-in by the sweep design,
matched samples-per-neuron ratios at every N, rather than a discovered
collapse.)

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
MEASURES = [("corr", "correlation"), ("E_rec", "excitatory recall"),
            ("E_prec", "excitatory precision")]


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

    fig, axes = plt.subplots(len(MEASURES), 1, figsize=(7, 11), squeeze=False)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.06,
                        hspace=0.30)

    for r, (key, label) in enumerate(MEASURES):
        ax = axes[r][0]
        for (N, (T, vals)), c in zip(sorted(data.items()), colors):
            ax.plot(T, vals[key], "o-", color=c, lw=1.9, ms=6, label=f"N = {N}")
        ax.set_xscale("log")
        ax.set(xlabel="recording length (ms)", ylabel=label, ylim=(0, 1.02))
        ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True)
        fs.despine(ax)
    axes[0][0].legend(fontsize=9, loc="lower right")

    title = args.title or ("Scaling: recovery is set by data per neuron "
                           "(matched AI regime, ~14 Hz, OLS)")
    fig.suptitle(title, fontsize=13, color=fs.INK, x=0.08, ha="left", y=0.965)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
