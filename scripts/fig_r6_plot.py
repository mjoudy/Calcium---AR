"""
R.6 (dynamical regime) — PLOT stage, local.

Does the network's dynamical state help or hurt connectivity inference? Three
N=12500 runs sit in different regimes, scored identically (analyze_run.py, OLS,
10% operating point) at matched recording lengths:

    canonical  (Fig 8B)  g=6, J=0.10, eta=4.0  -> 59 Hz, CV 0.86, sync 0.010
    low-rate             g=6, J=0.10, eta=1.5  -> 14 Hz, CV 0.62, sync 0.008
    tuned AI   (r4)      g=8, J=0.15, eta=2.6  -> 14 Hz, CV 1.06, sync 0.007

The two 14 Hz runs (low-rate vs tuned AI) isolate IRREGULARITY at a fixed rate;
canonical is the high-rate reference. NOTE the tuned-AI run also has larger J, so
the low-rate -> tuned-AI contrast changes CV and coupling together — this figure
is a descriptive regime comparison, not a single-variable causal claim.

    left  column : metric vs recording length (one line per regime)
    right column : the same metric at the longest recording, as labelled bars

rows = correlation and excitatory recall (OLS).

Usage:
  python scripts/fig_r6_plot.py --root ~/calcium_results/hpc_metrics --out figures/fig_R6
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

# (label, dir prefix, measured diagnostics, colour)
REGIMES = [
    ("canonical\n59 Hz, CV 0.86", "wrapup_n12500",   "#7a4fa3"),
    ("low-rate\n14 Hz, CV 0.62",  "wrapup_n12500lr", "#898781"),
    ("tuned AI\n14 Hz, CV 1.06",  "wrapup_n12500r4", "#2a78d6"),
]
T_LENGTHS = [1000, 2000, 5000]   # k-ms checkpoints common to all three
N = 12500
DT = 0.1
MEASURES = [("corr", "correlation"), ("E_rec", "excitatory recall")]


def load(root: Path, method="ols"):
    """-> {prefix: {'T':[ms], measure:[...]}} reading each metrics.csv."""
    out = {}
    for _, prefix, _c in REGIMES:
        Ts, vals = [], {k: [] for k, _ in MEASURES}
        for Tk in T_LENGTHS:
            f = root / f"{prefix}_T{Tk}k" / "metrics.csv"
            if not f.exists():
                continue
            with open(f) as fh:
                for row in csv.DictReader(fh):
                    if row["method"] == method:
                        Ts.append(Tk * 1000.0)
                        for k, _ in MEASURES:
                            vals[k].append(float(row[k]))
                        break
        if Ts:
            out[prefix] = {"T": np.array(Ts),
                           **{k: np.array(vals[k]) for k, _ in MEASURES}}
        else:
            print(f"[warn] no metrics for {prefix} (scored yet?)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/mjoudy/calcium_results/hpc_metrics")
    ap.add_argument("--out", default="figures/fig_R6")
    args = ap.parse_args()

    fs.apply_style()
    data = load(Path(args.root))
    if not data:
        raise SystemExit(f"no metrics.csv found under {args.root}")

    fig, axes = plt.subplots(len(MEASURES), 2, figsize=(12.5, 8), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.89, bottom=0.16,
                        hspace=0.30, wspace=0.22)

    for r, (key, label) in enumerate(MEASURES):
        axL, axR = axes[r][0], axes[r][1]
        bar_vals, bar_cols, bar_labels = [], [], []
        for lab, prefix, col in REGIMES:
            if prefix not in data:
                continue
            d = data[prefix]
            axL.plot((d["T"] / DT) / N, d[key], "o-", color=col, lw=2, ms=6,
                     label=lab.replace("\n", "  "))
            bar_vals.append(d[key][-1]); bar_cols.append(col)
            bar_labels.append(lab)
        axL.set_xscale("log")
        axL.set(xlabel="samples per neuron   T / N", ylabel=label, ylim=(0, 1.02))
        axL.grid(True, color=fs.GRID, lw=0.6); axL.set_axisbelow(True); fs.despine(axL)

        x = np.arange(len(bar_vals))
        axR.bar(x, bar_vals, color=bar_cols, width=0.62)
        for xi, v in zip(x, bar_vals):
            axR.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        axR.set_xticks(x); axR.set_xticklabels(bar_labels, fontsize=8.5)
        axR.set(ylabel=label, ylim=(0, 1.02))
        axR.grid(True, axis="y", color=fs.GRID, lw=0.6); axR.set_axisbelow(True)
        fs.despine(axR)
        if r == 0:
            axL.legend(fontsize=9, loc="lower right")
            axL.set_title("recovery vs data, per regime")
            tn_max = round(T_LENGTHS[-1] * 1000 / DT / N)
            axR.set_title(f"at the longest recording (T/N = {tn_max})")

    fig.suptitle("Dynamical regime and recovery at N=12500 (OLS): "
                 "does the regime change what can be inferred?",
                 fontsize=13, color=fs.INK, x=0.08, ha="left", y=0.965)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
