"""
Scaling result: recovery vs recording length at N=1250 and N=12500 (canonical
Brunel, AI regime), and the collapse onto samples-per-neuron (T/N).

Left panel  — ROC-AUC vs recording length (ms).
Right panel — the same points vs T/N = samples per neuron. If the curves collapse,
              the required recording length scales LINEARLY with network size.

Data provenance (OLS, seed 1):
  N=12500 canonical, eta=4  (~59 Hz) : results/wrapup_n12500_T{1000,2000,5000}k/metrics.csv
  N=12500 canonical, eta=1.5(~14 Hz) : results/wrapup_n12500lr_T{1000,2000,5000}k/metrics.csv
  N=1250  n1250ai   (~8 Hz, V_r=0)   : scripts/recovery_curve.py sweep
Recorded here so the figure can be redrawn without the cluster.

Usage:  python scripts/scaling_curve.py [--out fig.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DT = 0.1  # ms per sample

METRICS_DIR = Path("/home/mjoudy/calcium_results/hpc_metrics")

# N=12500 series are READ FROM the cluster-written metrics.csv files (prefix ->
# label/colour). N=1250 stays inline: its sweep predates the on-cluster scorer.
CSV_SERIES = [
    ("wrapup_n12500_T",   "N=12500 canonical, eta=4 (59 Hz)",   12500, "#2a78d6"),
    ("wrapup_n12500lr_T", "N=12500 canonical, eta=1.5 (14 Hz)", 12500, "#1baf7a"),
]
INLINE_SERIES = [
    ("N=1250 (8 Hz)", 1250, "#4a3aa7", [
        (50_000, 0.725, 0.300),
        (500_000, 0.961, 0.733),
        (1_000_000, 0.984, 0.830),
        (2_000_000, 0.993, 0.885),
    ]),
]


def load_series(method="ols"):
    """Read (recording_ms, roc_auc, E_rec) per series from metrics.csv files."""
    import csv as _csv
    out = []
    for prefix, label, N, colour in CSV_SERIES:
        pts = []
        for d in sorted(METRICS_DIR.glob(f"{prefix}*k")):
            f = d / "metrics.csv"
            if not f.exists():
                continue
            T_ms = float(d.name.rsplit("_T", 1)[1].rstrip("k")) * 1000.0
            with open(f) as fh:
                for row in _csv.DictReader(fh):
                    if row["method"] == method:
                        pts.append((T_ms, float(row["roc_auc"]), float(row["E_rec"])))
                        break
        if pts:
            out.append((label, N, colour, sorted(pts)))
        else:
            print(f"[warn] no CSVs for {prefix}* under {METRICS_DIR}")
    out.extend(INLINE_SERIES)
    return out

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/mjoudy/calcium_results/scaling_curve.png")
    ap.add_argument("--metric", default="auc", choices=["auc", "E_rec"])
    args = ap.parse_args()

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7"})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.85, bottom=0.14, wspace=0.24)

    for label, N, c, pts in load_series():
        T_ms = np.array([p[0] for p in pts], dtype=float)
        auc = np.array([p[1] for p in pts], dtype=float)
        erec = np.array([p[2] for p in pts], dtype=float)
        y = auc if args.metric == "auc" else erec
        samples_per_neuron = (T_ms / DT) / N

        a1.plot(T_ms, y, "o-", color=c, lw=2, ms=7, label=label)
        a2.plot(samples_per_neuron, y, "o-", color=c, lw=2, ms=7, label=label)

    ylab = "ROC-AUC" if args.metric == "auc" else "excitatory recall @10%"
    for ax, xlab, title in [
        (a1, "recording length (ms)", "vs recording length"),
        (a2, "samples per neuron   T/N", "vs samples per neuron (T/N)"),
    ]:
        ax.set_xscale("log")
        ax.set(xlabel=xlab, ylabel=ylab, title=title, ylim=(0, 1.02))
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    a1.legend(frameon=False, fontsize=9, loc="lower right")
    a2.annotate("curves collapse ->\nrequired recording ~ N",
                xy=(0.04, 0.10), xycoords="axes fraction", fontsize=9, color=MUTED)

    fig.suptitle("Connectivity recovery scales with samples per neuron "
                 "(canonical Brunel, AI)", fontsize=13, color=INK, x=0.07, ha="left")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
