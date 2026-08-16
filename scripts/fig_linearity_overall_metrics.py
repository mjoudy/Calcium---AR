"""
Overall connectivity-recovery performance, as a figure -- the companion to
fig_linearity_way2*.py's shared-input-confound lens. Answers a different,
more basic question: can each ground truth's estimator find real connections
at all? Motivated directly by the Hawkes investigation: a flat confound curve
only means something if the underlying detector is actually working (see
scripts/linearity_overall_metrics.py, whose numbers this plots).

Two panels, both with a chance-level reference line: AUC (0.5 = coin flip)
and top-10%-density precision (chance = the network's own true density,
~9.5%). Same estimator, same arms, same colours as fig_linearity_way2.py, so
this is meant to sit right next to it in a report.

Usage:
  python scripts/fig_linearity_overall_metrics.py --out figures/fig_linearity_overall_metrics
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import figstyle as fs
from calcium_ar.experiments.metrics import connectivity_metrics
from calcium_ar.solvers.from_moments import ols_from_moments

# Same arms, same colours as fig_linearity_way2.py -- deliberately kept in
# sync so the two figures read as one connected story.
SOURCES = [
    ("LIF (real)", "~/calcium_results/best_moments/n1250r4", "#2a78d6"),
    ("PIF pilot (tau_m x10)", str(ROOT / "results/wrapup_n1250pif_T100k/seed1"), "#e8a33d"),
    ("OU (exact linear)", "~/calcium_results/ou_moments/n1250_linear", "#c0392b"),
    ("PIF pilot (tau_m x100)", "~/calcium_results/wrapup_n1250pif100_T100k/seed1", "#b8860b"),
    ("Hawkes + calcium (linear)", "~/calcium_results/hawkes_moments/n1250_calcium_recal", "#2ca02c"),
]

# Rate/CV aren't derivable from Cxx/Cyx alone (need raw spike trains, not
# cached for every arm locally) -- these are measured, documented values from
# this project's own runs (wrapup_run.py NETS comments; PIF pilot probes;
# scripts/hawkes_ground_truth.py rate check + local ISI-CV measurement for
# Hawkes). OU has no rate/CV -- it's a continuous process, not a point
# process, so "firing rate" and "ISI" don't apply; shown as n/a rather than 0.
RATE_CV = {
    "LIF (real)": (14.8, 0.98),
    "PIF pilot (tau_m x10)": (13.7, 1.94),
    "PIF pilot (tau_m x100)": (14.8, 6.14),
    "OU (exact linear)": (None, None),
    "Hawkes + calcium (linear)": (14.5, 1.05),
}


def precision_at_density(A: np.ndarray, adj: np.ndarray, density: float = 0.10) -> float:
    N = adj.shape[0]
    off = ~np.eye(N, dtype=bool)
    aa = np.abs(A)
    truth = (adj.T != 0)
    tau_thr = np.quantile(aa[off], 1.0 - density)
    pred = aa > tau_thr
    tp = (pred & truth & off).sum(); fp = (pred & ~truth & off).sum()
    return tp / max(tp + fp, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/fig_linearity_overall_metrics")
    args = ap.parse_args()

    labels, colors, aucs, precisions, densities, rates, cvs = [], [], [], [], [], [], []
    for label, path, color in SOURCES:
        d = Path(path).expanduser()
        if not (d / "Cxx.npy").exists():
            print(f"[skip] {label}: no data at {path}")
            continue
        Cxx = np.load(d / "Cxx.npy"); Cyx = np.load(d / "Cyx.npy")
        adj = np.load(d / "adj_true.npy").astype(np.float64)
        np.fill_diagonal(adj, 0.0)
        A = ols_from_moments(Cxx, Cyx)
        m = connectivity_metrics.compute(adj_inferred=A, adj_true=adj)
        labels.append(label); colors.append(color)
        aucs.append(m["auc_roc"])
        precisions.append(precision_at_density(A, adj))
        densities.append(float((adj.T != 0).mean()))
        r, c = RATE_CV.get(label, (None, None))
        rates.append(r); cvs.append(c)
        print(f"{label:30s}  AUC={aucs[-1]:.3f}  precision={precisions[-1]:.3f}  "
              f"rate={r if r is not None else 'n/a'}  CV={c if c is not None else 'n/a'}")

    fs.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.0))
    fig.subplots_adjust(left=0.24, right=0.97, top=0.90, bottom=0.08,
                        wspace=0.15, hspace=0.35)
    (axAUC, axPrec), (axRate, axCV) = axes
    y = np.arange(len(labels))

    axAUC.barh(y, aucs, color=colors, height=0.6)
    axAUC.axvline(0.5, color="#888", lw=1.2, ls="--", zorder=0)
    axAUC.set(yticks=y, yticklabels=labels, xlim=(0, 1.0), xlabel="AUC (edge detection)",
              title="Detect real edges at all? (chance=0.5)")
    axAUC.invert_yaxis()
    fs.despine(axAUC)

    dens = np.mean(densities)
    axPrec.barh(y, precisions, color=colors, height=0.6)
    axPrec.axvline(dens, color="#888", lw=1.2, ls="--", zorder=0)
    axPrec.set(yticks=y, yticklabels=[], xlim=(0, 1.0), xlabel="precision @ top-10% density",
               title=f"Top-10% guesses correct? (chance={dens:.2f})")
    axPrec.invert_yaxis()
    fs.despine(axPrec)

    rates_plot = [r if r is not None else 0 for r in rates]
    axRate.barh(y, rates_plot, color=colors, height=0.6)
    for yi, r in zip(y, rates):
        if r is None:
            axRate.text(0.3, yi, "n/a (no spikes)", va="center", fontsize=8, color="#888")
    axRate.set(yticks=y, yticklabels=labels, xlabel="mean firing rate [Hz]",
               title="What regime is each arm actually in?")
    axRate.invert_yaxis()
    fs.despine(axRate)

    cvs_plot = [c if c is not None else 0 for c in cvs]
    axCV.barh(y, cvs_plot, color=colors, height=0.6)
    axCV.axvline(1.0, color="#888", lw=1.2, ls="--", zorder=0)
    for yi, c in zip(y, cvs):
        if c is None:
            axCV.text(0.3, yi, "n/a (no spikes)", va="center", fontsize=8, color="#888")
    axCV.set(yticks=y, yticklabels=[], xlabel="ISI coefficient of variation (CV)",
             title="How bursty/irregular is the firing? (Poisson=1)")
    axCV.invert_yaxis()
    fs.despine(axCV)

    fig.suptitle("Overall performance AND regime characteristics, N=1250\n"
                  "(companion to the shared-input-confound figures -- both detection "
                  "quality and firing regime matter for reading those curves honestly)",
                  fontsize=12, color=fs.INK)
    fs.save(fig, args.out)
    print(f"\nwrote {args.out}.pdf/.png")


if __name__ == "__main__":
    main()
