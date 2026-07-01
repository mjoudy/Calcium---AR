"""
Aggregate one parameter sweep into metric-vs-parameter curves (mean ± std over seeds).

Reads every <sweep_dir>/*/stages.csv the sweep produced, and for each metric plots
the swept parameter vs the metric, with one line per pipeline stage
(EN / EN+Dale / EN+Dale+balance) and a shaded seed error band. Also writes a
mean±std summary CSV for the paper tables.

Usage:
    python scripts/plot_sweeps.py <sweep_dir> [--out fig.png]
e.g. python scripts/plot_sweeps.py $(ws_find calcium_ar)/results/n1250_sweeps/lag
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ["pearson", "spearman", "auc_roc", "f1", "dale_type_accuracy", "ei_ratio"]
STAGES = ["EN", "EN+Dale", "EN+Dale+balance"]


def load_sweep(sweep_dir):
    files = sorted(glob.glob(os.path.join(sweep_dir, "*", "stages.csv")))
    if not files:
        raise SystemExit(f"no stages.csv found under {sweep_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df


def plot_sweep(sweep_dir, out=None):
    df = load_sweep(sweep_dir)
    sweep = str(df["sweep"].iloc[0])
    param = str(df["param"].iloc[0])
    logx = bool((df["value"] > 0).all())

    ncol = 3
    nrow = int(np.ceil(len(METRICS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
    axes = axes.ravel()

    for i, metric in enumerate(METRICS):
        ax = axes[i]
        for stage in STAGES:
            sub = df[df["stage"] == stage]
            if sub.empty:
                continue
            g = (sub.groupby("value")[metric]
                    .agg(["mean", "std"]).reset_index().sort_values("value"))
            sd = g["std"].fillna(0.0)
            ax.plot(g["value"], g["mean"], "-o", ms=3, label=stage)
            ax.fill_between(g["value"], g["mean"] - sd, g["mean"] + sd, alpha=0.2)
        ax.set(title=metric, xlabel=param, ylabel=metric)
        if logx:
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    for j in range(len(METRICS), len(axes)):
        axes[j].axis("off")

    n_seeds = df["seed"].nunique()
    fig.suptitle(f"N=1250 sweep: {sweep} ({param})   — mean ± std over {n_seeds} seeds",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = out or os.path.join(sweep_dir, f"sweep_{sweep}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)

    # summary table (mean ± std per stage/value) for the paper
    agg = (df.groupby(["stage", "value"])[METRICS]
             .agg(["mean", "std"]).round(4))
    agg_csv = os.path.join(sweep_dir, f"sweep_{sweep}_summary.csv")
    agg.to_csv(agg_csv)
    return out, agg_csv


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="metric-vs-parameter curves for one sweep")
    p.add_argument("sweep_dir")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    fig_path, csv_path = plot_sweep(a.sweep_dir, a.out)
    print("wrote", fig_path)
    print("wrote", csv_path)
