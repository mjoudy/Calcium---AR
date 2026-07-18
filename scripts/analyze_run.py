"""
Score a run ON THE CLUSTER and write a small CSV — so only kilobytes travel home.

At N=12500 each estimate is a 12500x12500 matrix (0.6-1.25 GB) and there are
~1.6e8 off-diagonal edges, which makes naive scoring both slow and memory-hungry
(it OOMs a laptop). This script:
  - loads ONE matrix at a time and frees it immediately,
  - scores every off-diagonal edge when the network is small enough,
    otherwise a large random sample (default 20M edges, which pins AUC/recall to
    ~3 decimals),
  - writes results/<run>/metrics.csv  (one row per seed x method).

Usage:
    python scripts/analyze_run.py --data $WS/results/wrapup_n12500ai_T1000k
    python scripts/analyze_run.py --data results/wrapup_n1250ai_T500k --density 0.10
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIELDS = ["run", "seed", "method", "n_scored", "sampled", "true_density",
          "corr", "roc_auc", "pr_ap", "auc_E", "auc_I",
          "E_rec", "E_prec", "I_rec", "I_prec", "none_rec"]


def edge_index(N, sample, rng):
    """Off-diagonal edge indices: all of them, or a random sample if too many."""
    n_off = N * (N - 1)
    if n_off <= sample:
        m = ~np.eye(N, dtype=bool)
        i, j = np.nonzero(m)
        return i, j, False
    i = rng.integers(0, N, sample)
    j = rng.integers(0, N, sample)
    k = i != j
    return i[k], j[k], True


def score(a, g, density):
    """Metrics for one estimate. a = inferred (signed), g = true (signed)."""
    aa = np.abs(a)
    yE, yI, yc = g > 0, g < 0, g != 0
    out = {}
    out["corr"] = float(np.corrcoef(a, g)[0, 1])
    out["roc_auc"] = float(roc_auc_score(yc, aa))
    out["pr_ap"] = float(average_precision_score(yc, aa))
    out["auc_E"] = float(roc_auc_score(yE, a)) if yE.any() else float("nan")
    out["auc_I"] = float(roc_auc_score(yI, -a)) if yI.any() else float("nan")
    # operating point: threshold |w| to the target predicted density
    tau = float(np.quantile(aa, 1.0 - density))
    conn = aa > tau if tau > 0 else aa > 0
    pE, pI = conn & (a > 0), conn & (a < 0)
    out["E_rec"] = float((pE & yE).sum() / max(yE.sum(), 1))
    out["E_prec"] = float((pE & yE).sum() / max(pE.sum(), 1))
    out["I_rec"] = float((pI & yI).sum() / max(yI.sum(), 1))
    out["I_prec"] = float((pI & yI).sum() / max(pI.sum(), 1))
    out["none_rec"] = float(((~conn) & (~yc)).sum() / max((~yc).sum(), 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir containing seed*/ subdirs")
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--sample", type=int, default=20_000_000,
                    help="max edges to score (larger networks are sampled)")
    ap.add_argument("--out", default=None, help="default: <data>/metrics.csv")
    args = ap.parse_args()

    data = Path(args.data)
    seeds = sorted([p for p in data.glob("seed*") if (p / "adj_true.npy").exists()])
    if not seeds:
        raise SystemExit(f"no seed*/ with adj_true.npy under {data}")
    out_path = Path(args.out) if args.out else data / "metrics.csv"
    rows = []

    for sd in seeds:
        rng = np.random.default_rng(0)          # same edges for every method
        adj = np.load(sd / "adj_true.npy", mmap_mode="r")
        N = adj.shape[0]
        i, j, sampled = edge_index(N, args.sample, rng)
        # convention: A[i,j] is compared against adj.T[i,j] = adj[j,i]
        g = np.asarray(adj[j, i], dtype=np.float64)
        del adj; gc.collect()
        true_density = float((g != 0).mean())

        for f in sorted(sd.glob("A_*.npy")):
            method = f.stem[2:]
            A = np.load(f, mmap_mode="r")
            a = np.asarray(A[i, j], dtype=np.float64)
            del A; gc.collect()
            r = score(a, g, args.density)
            r.update(run=data.name, seed=sd.name, method=method,
                     n_scored=len(g), sampled=int(sampled),
                     true_density=round(true_density, 5))
            rows.append(r)
            print(f"[{sd.name}] {method:9s} AUC={r['roc_auc']:.3f} "
                  f"AUC_E={r['auc_E']:.3f} AUC_I={r['auc_I']:.3f} "
                  f"E_rec={r['E_rec']:.3f} I_rec={r['I_rec']:.3f} corr={r['corr']:.3f}",
                  flush=True)
            del a; gc.collect()
        del g; gc.collect()

    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"\nwrote {out_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
