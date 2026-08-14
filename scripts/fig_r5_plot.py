"""
R.5 (regularization vs data) — PLOT stage, local.

Two figures from the r5_<net>.csv files:

  fig_R5          the CROSSOVER. metric vs samples-per-neuron T/N, one line per
                  estimator (each at its BEST lambda). Tests the hypothesis:
                  does regularization beat OLS when T/N is small, and does OLS
                  catch up as T/N grows?

  fig_R5_lambda   the lambda LANDSCAPE. metric vs lambda/lambda_max for the
                  elastic net, one line per T/N, with OLS as a dashed reference.
                  Shows the U-curve and how the optimum lambda shifts with data.

rows = correlation (threshold-free) and excitatory recall (the confounding-
limited frontier); inhibitory is solved everywhere so it is not shown.

Usage:
  python scripts/fig_r5_plot.py --root results/r5 --nets n1250r4 n12500r4 \
      --out figures/fig_R5
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

MEASURES = [("corr", "correlation"), ("E_rec", "excitatory recall"),
            ("pr_ap", "PR-AUC (connected vs. none)")]
METHOD_ORDER = ["ols", "ridge", "lasso", "en", "lassodale", "endale"]
METHOD_LABEL = {"ols": "OLS", "ridge": "Ridge", "lasso": "Lasso", "en": "EN",
                "lassodale": "Lasso+Dale", "endale": "EN+Dale"}
METHOD_COLOR = {"ols": "#111111", "ridge": "#2a78d6", "lasso": "#e08e2a",
                "en": "#3a9b52", "lassodale": "#b05fc8", "endale": "#e34948"}


def load(root: Path, net: str):
    rows = []
    with open(root / f"r5_{net}.csv") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def best_vs_TN(rows, method, key):
    """-> (TN[], best metric over lambda[]) for one estimator."""
    by_tn = {}
    for r in rows:
        if r["method"] != method:
            continue
        v = r[key]
        if v == "" or v != v:  # empty / nan guard
            try:
                v = float(v)
            except ValueError:
                continue
        else:
            v = float(v)
        if not np.isfinite(v):
            continue
        tn = float(r["TN"])
        by_tn[tn] = max(by_tn.get(tn, -np.inf), v)
    tns = sorted(by_tn)
    return np.array(tns), np.array([by_tn[t] for t in tns])


def fig_crossover(data, out):
    nets = list(data)
    fig, axes = plt.subplots(len(MEASURES), len(nets),
                             figsize=(6.4 * len(nets), 3.6 * len(MEASURES)), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.09,
                        hspace=0.28, wspace=0.2)
    for c, net in enumerate(nets):
        rows = data[net]
        N = int(rows[0]["N"])
        for r, (key, label) in enumerate(MEASURES):
            ax = axes[r][c]
            for m in METHOD_ORDER:
                tn, val = best_vs_TN(rows, m, key)
                if len(tn) == 0:
                    continue
                ax.plot(tn, val, "o-", color=METHOD_COLOR[m], lw=2.2 if m == "ols"
                        else 1.6, ms=6 if m == "ols" else 4.5,
                        zorder=5 if m == "ols" else 3,
                        label=METHOD_LABEL[m] if r == 0 and c == 0 else None)
            ax.set_xscale("log")
            ax.set(xlabel="samples per neuron   T / N", ylabel=label,
                   ylim=(0, 1.02))
            ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True)
            fs.despine(ax)
            if r == 0:
                ax.set_title(f"N = {N}")
    axes[0][0].legend(fontsize=9, loc="lower right", ncol=2)
    fig.suptitle("Recovery vs data per neuron, each estimator at its best λ "
                 "— does regularization lead at small T/N?",
                 fontsize=13, color=fs.INK, x=0.08, ha="left", y=0.965)
    fs.save(fig, out)


def fig_lambda(data, out, method="en"):
    nets = list(data)
    fig, axes = plt.subplots(len(MEASURES), len(nets),
                             figsize=(6.4 * len(nets), 3.6 * len(MEASURES)), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.09,
                        hspace=0.28, wspace=0.2)
    for c, net in enumerate(nets):
        rows = data[net]
        N = int(rows[0]["N"])
        tns = sorted({float(r["TN"]) for r in rows})
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(tns)))
        for r, (key, label) in enumerate(MEASURES):
            ax = axes[r][c]
            for tn, col in zip(tns, colors):
                pts = [(float(x["lam_over_max"]), float(x[key])) for x in rows
                       if x["method"] == method and float(x["TN"]) == tn
                       and x["lam_over_max"] != "" and np.isfinite(float(x[key]))]
                pts.sort()
                if pts:
                    lm = np.array([p[0] for p in pts]); mv = np.array([p[1] for p in pts])
                    ax.plot(lm, mv, "o-", color=col, lw=1.7, ms=4,
                            label=f"T/N = {tn:.0f}" if r == 0 and c == 0 else None)
                # OLS reference for this T/N
                ols = [float(x[key]) for x in rows if x["method"] == "ols"
                       and float(x["TN"]) == tn and np.isfinite(float(x[key]))]
                if ols:
                    ax.axhline(ols[0], color=col, ls=":", lw=1.1, zorder=1)
            ax.set_xscale("log")
            ax.set(xlabel="λ / λ_max", ylabel=label, ylim=(0, 1.02))
            ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True)
            fs.despine(ax)
            if r == 0:
                ax.set_title(f"N = {N}")
    axes[0][0].legend(fontsize=9, loc="lower left", title="solid = EN\ndotted = OLS")
    fig.suptitle("Elastic-net λ landscape (solid) vs the OLS reference (dotted), "
                 "one curve per recording length",
                 fontsize=13, color=fs.INK, x=0.08, ha="left", y=0.965)
    fs.save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/r5")
    ap.add_argument("--nets", nargs="+", default=["n1250r4", "n12500r4"])
    ap.add_argument("--out", default="figures/fig_R5")
    args = ap.parse_args()

    fs.apply_style()
    root = Path(args.root)
    data = {}
    for net in args.nets:
        f = root / f"r5_{net}.csv"
        if f.exists():
            data[net] = load(root, net)
        else:
            print(f"[warn] {f} not found, skipping")
    if not data:
        raise SystemExit(f"no r5_*.csv under {root}")

    fig_crossover(data, args.out)
    fig_lambda(data, args.out + "_lambda")


if __name__ == "__main__":
    main()
