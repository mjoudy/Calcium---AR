"""
R.8 (shared input vs directionality) — false-positive SPLIT violin, local.

Same underlying data/pipeline as fig_r8_violin.py, but the false-positive
bucket is no longer a single group. The original E_false/I_false classes only
check whether THIS direction (A[i,j]) is a true edge — they never look at
whether the REVERSE direction (A[j,i]) is. A[i,j] and A[j,i] are two
physically different synapses, so a "false positive" here could be:

  FF  genuinely no connection either way          (a real fake)
  FT  this direction is empty, but the REVERSE
      direction IS a true synapse                  (mirrored/leaked true
                                                     edge, not a pure fake)

True-edge violins (E_true/I_true) are left untouched; only the false-positive
side is split, six violins total. Needs fig_r8_compute.py's *_FF/*_FT fields
— re-run it (on the cluster, where A_ols.npy/adj_true.npy live) if the .npz
predates this split.

Usage:
  python scripts/fig_r8_violin_fpsplit.py \
      --data ~/calcium_results/r8/r8_wrapup_n12500r4_T5000k.npz \
      --out figures/fig_R8_violin_fpsplit_n12500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

# (stem, n-key, label, color, alpha, hatch) — hatch marks the "reverse IS
# real" group so it visually stands apart from the genuine-fake group.
LABELS = [
    ("E_true", "n_E_true", "true E", fs.C_E, 0.55, None),
    ("E_false_FF", "n_E_false_FF", "E false pos.\n(reverse empty too)", fs.C_E, 0.30, None),
    ("E_false_FT", "n_E_false_FT", "E false pos.\n(reverse IS real!)", fs.C_E, 0.30, "//"),
    ("I_true", "n_I_true", "true I", fs.C_I, 0.55, None),
    ("I_false_FF", "n_I_false_FF", "I false pos.\n(reverse empty too)", fs.C_I, 0.30, None),
    ("I_false_FT", "n_I_false_FT", "I false pos.\n(reverse IS real!)", fs.C_I, 0.30, "//"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--metric", choices=["asym", "asymc"], default="asym")
    ap.add_argument("--out", default="figures/fig_R8_violin_fpsplit")
    args = ap.parse_args()

    fs.apply_style()
    z = np.load(args.data, allow_pickle=True)
    N = int(z["N"])
    prefix = args.metric
    GROUPS = [(f"{prefix}_{stem}", nkey, lab, col, alpha, hatch)
              for stem, nkey, lab, col, alpha, hatch in LABELS]
    data = [(lab, col, alpha, hatch, z[key], (int(z[nkey]) if nkey in z.files else None))
            for key, nkey, lab, col, alpha, hatch in GROUPS
            if key in z.files and len(z[key]) > 0]
    if not data:
        raise SystemExit(f"no {prefix}_*_FF/_FT fields in {args.data} — re-run "
                          "the updated fig_r8_compute.py to produce them")

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.8))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.16)

    positions = list(range(1, len(data) + 1))
    vals = [v for _, _, _, _, v, _ in data]
    parts = ax.violinplot(vals, positions=positions, showmedians=True,
                           showextrema=False, widths=0.8)
    for i, (lab, col, alpha, hatch, v, n_true) in enumerate(data):
        body = parts["bodies"][i]
        body.set_facecolor(col)
        body.set_alpha(alpha if hatch else alpha + 0.2)
        body.set_edgecolor(col)
        if hatch:
            body.set_hatch(hatch); body.set_edgecolor(fs.INK); body.set_linewidth(0.7)
        n_label = f"n={n_true:,}" if n_true is not None else f"n≥{len(v):,}"
        ax.text(positions[i], 1.05, n_label, ha="center", va="bottom",
                fontsize=8.5, color=fs.MUTED)
    parts["cmedians"].set_color(fs.INK); parts["cmedians"].set_linewidth(1.4)
    for i, (lab, col, alpha, hatch, v, n_true) in enumerate(data):
        stat = np.mean(v) if args.metric == "asymc" else np.median(v)
        stat_lab = f" mean {stat:.2f}" if args.metric == "asymc" else f" {stat:.2f}"
        ax.text(positions[i], stat, stat_lab, ha="left", va="center",
                fontsize=8.5, color=fs.INK)

    ax.set_xticks(positions); ax.set_xticklabels([lab for lab, *_ in data], fontsize=8.5)
    if args.metric == "asymc":
        ylabel = "count-based asymmetry: is the reverse direction\nALSO predicted connected? (0=bidirectional, 1=one-directional)"
    else:
        ylabel = "directional asymmetry  |A[i,j]-A[j,i]| / (|A[i,j]|+|A[j,i]|)"
    ax.set(ylabel=ylabel, ylim=(0, 1.12))
    ax.grid(True, axis="y", color=fs.GRID, lw=0.6); ax.set_axisbelow(True)
    fs.despine(ax)
    ax.set_title(f"False positives split by reverse-direction truth, N={N} (single-lag OLS)\n"
                 "hatched = this \"false positive\" is actually a real synapse in the OTHER direction",
                 fontsize=11.5, color=fs.INK)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
