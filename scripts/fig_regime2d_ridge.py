"""
2-D regime probe — summary panel.

The heatmaps show the whole (g, eta) plane; this reduces them to the only line
that matters: the ISO-RATE ridge, i.e. for each g the eta that gives the target
firing rate. Along that ridge we ask how irregular (ISI CV) and how asynchronous
(pairwise synchrony) the network is.

The point: rate and irregularity are NOT in conflict. Holding the rate fixed,
raising g (or J) raises CV and LOWERS synchrony — both move toward textbook AI.
The ladder's CV ~0.63 was a property of the g=6, J=0.316 corner, not a ceiling.

Local plotting only, reads the probe CSVs.

Usage:
  python scripts/fig_regime2d_ridge.py --root ~/calcium_results/regime2d \
      --out figures/fig_regime2d_ridge
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


def ridge(rows, target):
    """For each g: the eta hitting `target` Hz, and CV/sync interpolated there."""
    out = []
    for g in sorted({float(r["g"]) for r in rows}):
        sub = sorted((r for r in rows if float(r["g"]) == g),
                     key=lambda r: float(r["eta"]))
        eta = np.array([float(r["eta"]) for r in sub])
        rate = np.array([float(r["rate"]) for r in sub])
        cv = np.array([float(r["cv"]) for r in sub])
        sy = np.array([float(r["sync"]) for r in sub])
        k = np.flatnonzero((rate[:-1] - target) * (rate[1:] - target) <= 0)
        if len(k):                                   # bracket -> interpolate
            i = int(k[0])
            f = (target - rate[i]) / (rate[i + 1] - rate[i] + 1e-12)
            out.append((g, eta[i] + f * (eta[i + 1] - eta[i]),
                        cv[i] + f * (cv[i + 1] - cv[i]),
                        sy[i] + f * (sy[i + 1] - sy[i])))
        else:                                        # no crossing -> nearest
            i = int(np.argmin(abs(rate - target)))
            out.append((g, eta[i], cv[i], sy[i]))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/mjoudy/calcium_results/regime2d")
    ap.add_argument("--out", default="figures/fig_regime2d_ridge")
    ap.add_argument("--target-rate", type=float, default=14.0)
    ap.add_argument("--target-cv", type=float, default=1.0)
    args = ap.parse_args()

    fs.apply_style()
    files = sorted(Path(args.root).glob("regime2d_N*_j*.csv"),
                   key=lambda p: float(p.stem.split("_j")[1]))
    if not files:
        raise SystemExit(f"no probe CSVs under {args.root}")
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(files)))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.14, wspace=0.24)
    N = None
    for f, c in zip(files, colors):
        rows = list(csv.DictReader(open(f)))
        N, J, js = int(rows[0]["N"]), float(rows[0]["J"]), float(f.stem.split("_j")[1])
        R = ridge(rows, args.target_rate)
        lab = f"J = {J:.3f}  (x{js:g})"
        axes[0].plot(R[:, 0], R[:, 2], "o-", color=c, lw=2, ms=6, label=lab)
        axes[1].plot(R[:, 0], R[:, 3], "o-", color=c, lw=2, ms=6, label=lab)

    axes[0].axhline(args.target_cv, color="#e34948", ls="--", lw=1.5)
    axes[0].text(0.02, args.target_cv, f" textbook AI: CV = {args.target_cv:g}",
                 transform=axes[0].get_yaxis_transform(), va="bottom",
                 color="#e34948", fontsize=9.5)
    axes[0].set(xlabel="g  (inhibition dominance)", ylabel="ISI CV")
    axes[0].set_title(f"irregularity at fixed rate ({args.target_rate:g} Hz)")
    axes[1].set(xlabel="g  (inhibition dominance)", ylabel="pairwise synchrony")
    axes[1].set_title("...and the network gets MORE asynchronous too")
    for ax in axes:
        ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)
    axes[0].legend(fontsize=9, loc="upper left")

    fig.suptitle(f"N={N}: holding the firing rate at {args.target_rate:g} Hz, "
                 "stronger inhibition buys irregularity for free",
                 fontsize=12.5, color=fs.INK, x=0.07, ha="left", y=0.955)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
