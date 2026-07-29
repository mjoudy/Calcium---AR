"""
R.8 phase 2 (multiple lags) — PLOT, local.

    A  timing SIGNATURE: class-averaged |A_lag| vs lag. True edges should peak at
       the synaptic delay; false positives (shared input) should peak at the
       shortest lag (instantaneous). This is the timing axis on its own.
    B  the TWO AXES: directional asymmetry (symmetry axis, phase 1) vs peak lag
       (timing axis, phase 2), true edges vs false positives. Do they separate on
       each axis independently?
    C  VALUE-ADD: excitatory precision-recall when filtering by symmetry only,
       by timing only, or by both — does timing buy anything beyond symmetry?

Usage:
  python scripts/fig_r8b_plot.py --data ~/calcium_results/r8b/r8b_n12500_r4.npz \
      --out figures/fig_R8b_n12500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs


def pr_curve(pred_mask, true_mask, score, thresholds):
    """precision & recall as `score` is thresholded upward among predicted."""
    nT = max(true_mask.sum(), 1)
    prec = np.zeros(len(thresholds)); rec = np.zeros(len(thresholds))
    for k, th in enumerate(thresholds):
        keep = pred_mask & (score >= th)
        tp = (keep & true_mask).sum()
        prec[k] = tp / max(keep.sum(), 1); rec[k] = tp / nT
    return prec, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="figures/fig_R8b")
    args = ap.parse_args()

    fs.apply_style()
    z = np.load(args.data, allow_pickle=False)
    N = int(z["N"]); lags = z["lags_ms"]; delay = float(z["delay_ms"])
    lag_est = float(z["lag_est_ms"])

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.subplots_adjust(left=0.05, right=0.985, top=0.85, bottom=0.14, wspace=0.28)

    # A: timing signature
    a0 = ax[0]
    for key, lab, col in [("prof_Etrue", "true excitatory edges", fs.C_E),
                          ("prof_Efalse", "excitatory false positives", "#c0392b")]:
        if key in z.files:
            v = z[key]; a0.plot(lags, v / np.nanmax(v), "o-", color=col, lw=2, ms=5, label=lab)
    a0.axvline(delay, color=fs.INK, ls=":", lw=1.3)
    a0.text(delay, 1.02, f" synaptic delay {delay:g}ms", fontsize=8.5, color=fs.INK)
    a0.set(xlabel="lag (ms)", ylabel="mean |A(lag)|  (normalised)", ylim=(0, 1.1))
    a0.grid(True, color=fs.GRID, lw=0.6); a0.set_axisbelow(True); fs.despine(a0)
    a0.legend(fontsize=9, loc="upper right")
    a0.set_title("A. timing signature: where does the coupling peak?")

    # per-pair arrays
    peak = z["peak_lag_ms"]; asym = z["asym"]; a_est = z["a_est"]
    g = z["g_sign"]; pred = z["pred"].astype(bool)
    predE = pred & (a_est > 0); trueE = g > 0; ynone = g == 0

    # B: two axes
    a1 = ax[1]
    jit = lambda n: (np.random.default_rng(0).random(n) - 0.5) * (lags[1] - lags[0]) * 0.6
    for m, lab, col in [(predE & trueE, "true E edges", fs.C_E),
                        (predE & ynone, "E false positives", "#c0392b")]:
        idx = np.flatnonzero(m)
        if len(idx) > 4000:
            idx = np.random.default_rng(1).choice(idx, 4000, replace=False)
        a1.scatter(asym[idx], peak[idx] + jit(len(idx)), s=6, alpha=0.25, color=col,
                   label=lab, edgecolors="none")
    a1.axhline(delay, color=fs.INK, ls=":", lw=1.2)
    a1.set(xlabel="directional asymmetry  (symmetry axis)",
           ylabel="peak lag, ms  (timing axis)", xlim=(0, 1))
    a1.grid(True, color=fs.GRID, lw=0.6); a1.set_axisbelow(True); fs.despine(a1)
    a1.legend(fontsize=9, loc="upper left", markerscale=2)
    a1.set_title("B. two independent axes")

    # C: value-add — precision/recall under three filters
    a2 = ax[2]
    asym_thr = np.linspace(0, 1, 41)
    # timing score: closeness of peak lag to the delay (1 at delay, 0 far away)
    tscore = np.exp(-np.abs(peak - delay) / max(lag_est - 0.0, 1.0))
    t_thr = np.linspace(0, 1, 41)
    pS, rS = pr_curve(predE, trueE, asym, asym_thr)
    pT, rT = pr_curve(predE, trueE, tscore, t_thr)
    # combined: rank by product of the two scores
    comb = asym * tscore
    pC, rC = pr_curve(predE, trueE, comb, np.linspace(0, comb.max() + 1e-9, 41))
    a2.plot(rS, pS, "-", color="#2a78d6", lw=2, label="symmetry only")
    a2.plot(rT, pT, "-", color="#e08e2a", lw=2, label="timing only")
    a2.plot(rC, pC, "-", color="#3a9b52", lw=2.4, label="symmetry × timing")
    a2.scatter([rS[0]], [pS[0]], color=fs.INK, zorder=6)
    a2.annotate("baseline", (rS[0], pS[0]), fontsize=8.5, color=fs.INK,
                xytext=(4, -10), textcoords="offset points")
    a2.set(xlabel="excitatory recall", ylabel="excitatory precision",
           xlim=(0, max(rS[0], rT[0]) * 1.05), ylim=(0, 1.02))
    a2.grid(True, color=fs.GRID, lw=0.6); a2.set_axisbelow(True); fs.despine(a2)
    a2.legend(fontsize=9, loc="lower left")
    a2.set_title("C. does timing add to symmetry?")

    fig.suptitle(f"Shared input, two signatures at N={N}: direction (symmetry) and "
                 "timing (multi-lag) are separate axes",
                 fontsize=13, color=fs.INK, x=0.05, ha="left", y=0.965)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
