"""
What fraction of an arm's false positives are actually explained by shared-
input exposure, versus baseline error the method would have anyway?

Method: take the false-positive RATE (fraction of genuine non-edges wrongly
predicted "connected") among pairs with the LEAST shared presynaptic drivers
as the baseline -- the error rate you'd expect even with zero confound. If
that baseline rate applied to ALL non-edges regardless of driver count, you'd
expect baseline_rate * n_nonedges false positives. The excess over that
(actual - expected) is the portion of false positives specifically
attributable to shared-input exposure.

    attributable %  =  (actual_FP - baseline_rate * n_nonedges) / actual_FP * 100

Companion to fig_linearity_way2_rate.py (same rate framework, same arms) --
that figure shows the SHAPE of the confound; this gives one summary NUMBER per
arm: how much of the false-positive problem is this specific mechanism,
versus everything else (estimation noise, other confounds, etc.)?

Usage:
  python scripts/linearity_confound_attribution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SOURCES = [
    ("LIF", "~/calcium_results/best_moments/n1250r4"),
    ("PIF pilot (tau_m x10)", str(ROOT / "results/wrapup_n1250pif_T100k/seed1")),
    ("PIF pilot (tau_m x100)", "~/calcium_results/wrapup_n1250pif100_T100k/seed1"),
    ("Hawkes", "~/calcium_results/hawkes_moments/n1250_calcium_recal"),
]


def attribution(data_dir, density, min_n):
    d = Path(data_dir).expanduser()
    Cxx = np.load(d / "Cxx.npy"); Cyx = np.load(d / "Cyx.npy")
    adj = np.load(d / "adj_true.npy").astype(np.float64)
    N = Cxx.shape[0]; np.fill_diagonal(adj, 0.0)

    A = Cyx @ np.linalg.inv(Cxx + 1e-9 * np.eye(N))
    np.fill_diagonal(A, 0.0)
    aa = np.abs(A)
    off = ~np.eye(N, dtype=bool)
    tau = float(np.quantile(aa[off], 1.0 - density))
    pred = (aa > tau) & off
    truth = (adj.T != 0) & off
    nonedge = off & ~truth

    ia, ib = np.nonzero(nonedge)
    is_fp = pred[ia, ib]
    Bc = (adj != 0)
    driver_count = (Bc[:, ia] & Bc[:, ib]).sum(0)

    actual_fp = int(is_fp.sum())
    n_nonedges = len(ia)

    # baseline: lowest driver-count bin with enough pairs to trust (same
    # min_n convention as fig_linearity_way2_rate.py)
    vals = np.arange(driver_count.min(), driver_count.max() + 1)
    baseline_rate = None
    for v in vals:
        m = driver_count == v
        if m.sum() >= min_n:
            baseline_rate = is_fp[m].mean()
            baseline_driver_count = v
            break
    if baseline_rate is None:
        return None

    expected_fp = baseline_rate * n_nonedges
    excess = actual_fp - expected_fp
    pct_attributable = 100.0 * excess / actual_fp if actual_fp > 0 else float("nan")
    return dict(actual_fp=actual_fp, n_nonedges=n_nonedges, baseline_rate=baseline_rate,
                baseline_driver_count=int(baseline_driver_count), expected_fp=expected_fp,
                excess=excess, pct_attributable=pct_attributable)


def main():
    density, min_n = 0.10, 100
    hdr = (f"{'arm':30s} {'actual FP':>10s} {'baseline%':>10s} {'expected':>9s} "
           f"{'excess':>8s} {'% attributable':>15s}")
    print(hdr); print("-" * len(hdr))
    for label, path in SOURCES:
        if not (Path(path).expanduser() / "Cxx.npy").exists():
            print(f"[skip] {label}: no data")
            continue
        r = attribution(path, density, min_n)
        if r is None:
            print(f"[skip] {label}: no bin with >= {min_n} pairs")
            continue
        print(f"{label:30s} {r['actual_fp']:10d} {r['baseline_rate']*100:9.2f}% "
              f"{r['expected_fp']:9.1f} {r['excess']:8.1f} {r['pct_attributable']:14.1f}%")
    print(f"\nbaseline = false-positive rate at the lowest driver-count bin with "
          f">= {min_n} pairs (min shared-input exposure available in the data).")
    print("% attributable = (actual FP - baseline-rate-implied FP) / actual FP.")


if __name__ == "__main__":
    main()
