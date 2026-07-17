"""
Data-length recovery curve for the clean-AI regime (n1250ai): does more
recording recover inference where correlations are weak?

Plots ROC-AUC and excitatory recall vs recording length (50k -> 2M ms), one line
per method, seed error bars. Reads the streaming sweep outputs plus the 50k
in-memory baseline.

Usage:  python scripts/recovery_curve.py [--out fig.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from calcium_ar.experiments import thresholding as T

CR = "/home/mjoudy/calcium_results"
RUNS = [(50_000, f"{CR}/wrapup_n1250ai"),
        (500_000, f"{CR}/wrapup_n1250ai_T500k"),
        (1_000_000, f"{CR}/wrapup_n1250ai_T1000k"),
        (2_000_000, f"{CR}/wrapup_n1250ai_T2000k")]
METHODS = [("ols", "OLS", "#2a78d6"), ("en", "EN", "#1baf7a"),
           ("endale", "EN+Dale", "#4a3aa7")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def metric_vs_T(key, metric):
    xs, mean, std = [], [], []
    for Tms, d in RUNS:
        sds = [s for s in sorted(Path(d).glob("seed*")) if (s / f"A_{key}.npy").exists()]
        if not sds:
            continue
        vals = [T.score_all(np.load(s / f"A_{key}.npy"),
                            np.load(s / "adj_true.npy"), density=0.10)[metric] for s in sds]
        xs.append(Tms); mean.append(np.nanmean(vals)); std.append(np.nanstd(vals))
    return np.array(xs), np.array(mean), np.array(std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{CR}/recovery_curve.png")
    args = ap.parse_args()

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7"})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.13, wspace=0.24)

    for ax, metric, title, yl in [(a1, "roc_auc", "detection: ROC-AUC", "ROC-AUC"),
                                  (a2, "E_r", "excitatory recall @10%", "excitatory recall")]:
        for key, lab, c in METHODS:
            x, m, s = metric_vs_T(key, metric)
            ax.errorbar(x, m, yerr=s, marker="o", ms=6, lw=2, color=c, label=lab,
                        capsize=3, elinewidth=1)
        ax.set_xscale("log")
        ax.set_xticks([r[0] for r in RUNS])
        ax.set_xticklabels(["50k", "500k", "1M", "2M"])
        ax.set(xlabel="recording length (ms)", ylabel=yl, title=title, ylim=(0, 1.02))
        if metric == "roc_auc":
            ax.axhline(0.5, color=MUTED, ls=":", lw=1)
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    a1.legend(frameon=False, fontsize=9, loc="lower right")

    fig.suptitle("Clean-AI regime (N=1250): does more data recover inference?  "
                 "OLS yes, regularized no", fontsize=13, color=INK, x=0.07, ha="left")
    fig.savefig(args.out, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
