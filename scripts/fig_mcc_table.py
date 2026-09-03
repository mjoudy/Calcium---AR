"""
MCC across network size and dynamical regime — table + heatmap, local.

Reads the metrics.csv files written by scripts/analyze_run.py (which now carries
the MCC family: mcc3, mcc_pres, mcc_E, mcc_I, mcc_type, mcc_neuron) for the
wrap-up runs, aggregates over seeds, and renders:

  - results/fig_data/mcc_table.csv   tidy (N, regime, T, method) x 6 MCC columns
  - figures/fig_MCC_size_regime.{pdf,png}
        top    : heatmap  rows = (N, regime)  cols = 6 MCC flavours   [--method]
        bottom : MCC_3class and MCC_I vs N, one line per regime

Run names are parsed as  wrapup_n<N><fam>_T<T>k :
    fam ""  -> canonical (g=6, eta=4)      fam "lr" -> low-rate (g=6, eta=1.5)
    fam "r4"-> tuned-AI  (g=8)             fam "pif"-> PIF (near-linear)

Usage:
  python scripts/fig_mcc_table.py --root results --method ols --tpick longest
  python scripts/fig_mcc_table.py --root results --tpick matched-tn --tn 4000
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

MCCS = ["mcc3", "mcc_pres", "mcc_E", "mcc_I", "mcc_type", "mcc_neuron"]
MCC_LABEL = ["3-class\nE/none/I", "2-class\nedge/none", "E vs\nrest",
             "I vs\nrest", "sign\nE vs I", "neuron\ntype"]
REGIME = {"": "canonical", "lr": "low-rate", "r4": "tuned-AI", "pif": "PIF"}
REGIME_ORDER = ["canonical", "low-rate", "tuned-AI", "PIF"]
RUN_RE = re.compile(r"wrapup_n(\d+)([a-z0-9]*?)_T(\d+)k$")


def collect(root: Path, families: set[str]):
    """rows keyed (N, regime, T_k, method) -> list of per-seed dicts."""
    rows = defaultdict(list)
    for csv_path in sorted(root.glob("wrapup_n*/metrics.csv")):
        m = RUN_RE.search(csv_path.parent.name)
        if not m:
            continue
        N, fam, Tk = int(m.group(1)), m.group(2), int(m.group(3))
        if fam not in families:
            continue
        regime = REGIME.get(fam, fam)
        with open(csv_path) as fh:
            for r in csv.DictReader(fh):
                if not r.get("mcc3"):          # not re-scored yet — skip quietly
                    continue
                key = (N, regime, Tk, r["method"])
                rows[key].append({k: float(r[k]) for k in MCCS if r.get(k) not in (None, "")})
    return rows


def pick_T(rows, tpick, tn_target):
    """For each (N, regime, method) keep one T: longest, or nearest target T/N."""
    best = {}
    for (N, regime, Tk, method), seeds in rows.items():
        k = (N, regime, method)
        if tpick == "longest":
            score = Tk
            better = k not in best or Tk > best[k][0]
        else:                                  # matched-tn: minimise |T/N - target|
            tn = (Tk * 1e3 / 0.1) / N
            score = -abs(tn - tn_target)
            better = k not in best or score > best[k][0]
        if better:
            best[k] = (score, Tk, seeds)
    out = {}
    for (N, regime, method), (_, Tk, seeds) in best.items():
        agg = {mc: (float(np.mean([s[mc] for s in seeds if mc in s])),
                    float(np.std([s[mc] for s in seeds if mc in s])))
               for mc in MCCS}
        out[(N, regime, method)] = (Tk, len(seeds), agg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--method", default="ols", help="method for the heatmap panel")
    ap.add_argument("--tpick", choices=["longest", "matched-tn"], default="longest")
    ap.add_argument("--tn", type=float, default=4000.0, help="target T/N for matched-tn")
    ap.add_argument("--families", default=",".join(REGIME),
                    help="comma list of run families to include (default all known)")
    ap.add_argument("--out", default="figures/fig_MCC_size_regime")
    args = ap.parse_args()

    fams = set(f.strip() for f in args.families.split(","))   # "" == canonical
    rows = collect(Path(args.root), fams)
    if not rows:
        raise SystemExit("no re-scored metrics.csv found (need the mcc3 column). "
                         "Run scripts/analyze_run.py on the wrap-up dirs first.")
    picked = pick_T(rows, args.tpick, args.tn)

    # ---- tidy table ---------------------------------------------------------- #
    fs.apply_style()
    tdir = Path("results/fig_data"); tdir.mkdir(parents=True, exist_ok=True)
    tpath = tdir / "mcc_table.csv"
    with open(tpath, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N", "regime", "T_k", "n_seed", "method"]
                   + [f"{mc}_mean" for mc in MCCS] + [f"{mc}_std" for mc in MCCS])
        for (N, regime, method), (Tk, ns, agg) in sorted(picked.items()):
            w.writerow([N, regime, Tk, ns, method]
                       + [f"{agg[mc][0]:.4f}" for mc in MCCS]
                       + [f"{agg[mc][1]:.4f}" for mc in MCCS])
    print(f"wrote {tpath}")

    # printed markdown for the chosen method
    sub = {(N, rg): agg for (N, rg, mth), (_, _, agg) in picked.items() if mth == args.method}
    print(f"\nMCC — method={args.method}, T={args.tpick}\n")
    print("| N | regime | " + " | ".join(l.replace("\n", " ") for l in MCC_LABEL) + " |")
    print("|---|---|" + "---|" * len(MCCS))
    order = sorted(sub, key=lambda k: (k[0], REGIME_ORDER.index(k[1]) if k[1] in REGIME_ORDER else 9))
    for (N, rg) in order:
        agg = sub[(N, rg)]
        print(f"| {N} | {rg} | " + " | ".join(f"{agg[mc][0]:.3f}" for mc in MCCS) + " |")

    # ---- figure ------------------------------------------------------------- #
    labels = [f"N={N}\n{rg}" for (N, rg) in order]
    M = np.array([[sub[k][mc][0] for mc in MCCS] for k in order])
    fig = plt.figure(figsize=(10, 3.4 + 0.42 * len(order)))
    gs = fig.add_gridspec(2, 1, height_ratios=[len(order) + 1, 4], hspace=0.55)

    ax = fig.add_subplot(gs[0])
    im = ax.imshow(M, cmap=fs.BLUE_SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(MCCS))); ax.set_xticklabels(MCC_LABEL, fontsize=8.5)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(labels, fontsize=8.5)
    for i in range(len(order)):
        for j in range(len(MCCS)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.55 else fs.INK)
    ax.set_title(f"MCC by network size and regime  (method: {args.method.upper()}, "
                 f"{args.tpick} recording)", fontsize=10.5)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="MCC")
    fs.despine(ax, keep=())

    ax2 = fig.add_subplot(gs[1])
    regimes = sorted({rg for _, rg in order},
                     key=lambda r: REGIME_ORDER.index(r) if r in REGIME_ORDER else 9)
    for rg in regimes:
        Ns = sorted(N for (N, rg2) in order if rg2 == rg)
        y3 = [sub[(N, rg)]["mcc3"][0] for N in Ns]
        yI = [sub[(N, rg)]["mcc_I"][0] for N in Ns]
        line, = ax2.plot(Ns, y3, "o-", label=f"{rg} · 3-class")
        ax2.plot(Ns, yI, "s--", color=line.get_color(), alpha=0.6, label=f"{rg} · I-vs-rest")
    ax2.set_xscale("log"); ax2.set_xlabel("network size N"); ax2.set_ylabel("MCC")
    ax2.set_ylim(0, 1); ax2.legend(ncol=2, fontsize=8)
    ax2.set_title("size trend, per regime", fontsize=10.5)
    fs.despine(ax2)

    fs.save(fig, args.out)
    print(f"wrote {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
