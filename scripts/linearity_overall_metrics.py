"""
Overall connectivity-recovery performance for every arm in the linearity
ladder (LIF, PIF x10/x100, OU, Hawkes) -- NOT the shared-input false-positive
lens fig_linearity_way2.py uses, but the basic question underneath it: can
plain OLS find the real edges at all in this arm's data? Motivated by the
Hawkes result -- its flat false-positive curve turned out to mean "barely
finds anything," not "linearity suppresses the confound" -- so before reading
too much into any arm's false-positive shape, check this table first.

Same estimator as fig_linearity_way2.py (plain OLS from Cxx/Cyx, single seed,
100% observed), scored with the project's standard metrics registry
(calcium_ar.experiments.metrics.connectivity_metrics) plus a top-10%-density
precision number to connect back to that analysis.

Usage:
  python scripts/linearity_overall_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calcium_ar.experiments.metrics import connectivity_metrics
from calcium_ar.solvers.from_moments import ols_from_moments

SOURCES = [
    ("LIF (real)", "~/calcium_results/best_moments/n1250r4"),
    ("PIF pilot (tau_m x10)", str(ROOT / "results/wrapup_n1250pif_T100k/seed1")),
    ("PIF pilot (tau_m x100)", "~/calcium_results/wrapup_n1250pif100_T100k/seed1"),
    ("OU (exact linear)", "~/calcium_results/ou_moments/n1250_linear"),
    ("Hawkes (linear, real spikes)", "~/calcium_results/hawkes_moments/n1250_calcium_recal"),
]


def precision_at_density(A: np.ndarray, adj: np.ndarray, density: float = 0.10) -> float:
    """Same top-density-threshold precision used throughout the project's
    wrap-up evaluation and fig_linearity_way2.py's sanity checks."""
    N = adj.shape[0]
    off = ~np.eye(N, dtype=bool)
    aa = np.abs(A)
    truth = (adj.T != 0)                    # adj_true.T convention, see metrics.py
    tau_thr = np.quantile(aa[off], 1.0 - density)
    pred = aa > tau_thr
    tp = (pred & truth & off).sum(); fp = (pred & ~truth & off).sum()
    return tp / max(tp + fp, 1)


def main():
    rows = []
    for label, path in SOURCES:
        d = Path(path).expanduser()
        if not (d / "Cxx.npy").exists():
            print(f"[skip] {label}: no data at {path}")
            continue
        Cxx = np.load(d / "Cxx.npy"); Cyx = np.load(d / "Cyx.npy")
        adj = np.load(d / "adj_true.npy").astype(np.float64)
        np.fill_diagonal(adj, 0.0)
        A = ols_from_moments(Cxx, Cyx)
        m = connectivity_metrics.compute(adj_inferred=A, adj_true=adj)
        m["precision"] = precision_at_density(A, adj)
        density = float((adj.T != 0).mean())
        rows.append((label, density, m))

    hdr = (f"{'arm':30s} {'density':>8s} {'pearson':>8s} {'spearman':>9s} "
           f"{'auc_roc':>8s} {'precision':>10s}   (chance precision = density)")
    print(hdr)
    print("-" * (len(hdr) - 32))
    for label, density, m in rows:
        print(f"{label:30s} {density:8.3f} {m['pearson']:8.3f} {m['spearman']:9.3f} "
              f"{m['auc_roc']:8.3f} {m['precision']:10.3f}")


if __name__ == "__main__":
    main()
