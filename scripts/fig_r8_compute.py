"""
R.8 (shared input vs directionality) — COMPUTE stage, core.

Shared input fabricates SYMMETRIC edges: if a hidden neuron drives both i and j,
the single-lag estimate tends to make A[i,j] ~ A[j,i] (both predict each other).
A TRUE synaptic edge j->i is DIRECTIONAL: A[i,j] large, A[j,i] ~ 0. So an
asymmetry score should separate real edges from shared-input false positives:

    asym[i,j] = |A[i,j] - A[j,i]| / (|A[i,j]| + |A[j,i]| + eps)   in [0, 1]
                ~1  one-directional (looks real)
                ~0  symmetric       (looks like shared-input fake)

No new simulation: reads the already-saved A_ols.npy + adj_true.npy.

Two outputs, matching the two questions:
  (A) DIAGNOSIS — asym distribution for true edges vs false positives.
  (B) EFFECT/FIX — remove predicted edges below an asym threshold and re-score;
      does excitatory precision recover, and at what recall cost?

Convention (matches analyze_run): A[i,j] is edge j->i, true value adj.T[i,j].

Writes  <out-dir>/r8_<tag>.npz  and  .csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from analyze_run import edge_index
BASE = Path(os.environ.get("CALCIUM_AR_WORKDIR", ROOT))
EPS = 1e-12


def find_seed_dir(root, net):
    """Longest checkpoint's seed1 dir with A_ols + adj_true."""
    pref = f"wrapup_{net}_T"
    cks = sorted(Path(root).glob(f"{pref}*k"),
                 key=lambda p: int(p.name.rsplit("_T", 1)[1].rstrip("k")))
    for d in reversed(cks):
        sd = d / "seed1"
        if (sd / "A_ols.npy").exists() and (sd / "adj_true.npy").exists():
            return sd, d.name
    raise SystemExit(f"no A_ols.npy+adj_true.npy under {root}/{pref}*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="seed dir (A_ols.npy, adj_true.npy)")
    ap.add_argument("--root", default=str(BASE / "results"))
    ap.add_argument("--net", default=None, help="preset stem; picks longest checkpoint")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--sample", type=int, default=20_000_000)
    ap.add_argument("--n-thr", type=int, default=41, help="asym-filter sweep points")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.data:
        sd = Path(args.data); tag = args.tag or sd.parent.name
    else:
        if not args.net:
            raise SystemExit("give --data or --net")
        sd, name = find_seed_dir(args.root, args.net)
        tag = args.tag or name
    print(f"reading {sd}")

    A = np.load(sd / "A_ols.npy", mmap_mode="r")
    adj = np.load(sd / "adj_true.npy", mmap_mode="r")
    N = A.shape[0]
    i, j, sampled = edge_index(N, args.sample, np.random.default_rng(0))
    a = np.asarray(A[i, j], dtype=np.float64)       # inferred j->i
    a_rev = np.asarray(A[j, i], dtype=np.float64)   # inferred i->j
    g = np.asarray(adj[j, i], dtype=np.float64)     # true j->i
    del A, adj
    aa = np.abs(a)
    asym = np.abs(a - a_rev) / (np.abs(a) + np.abs(a_rev) + EPS)

    # operating point: top-density |a| are the predicted edges
    tau = float(np.quantile(aa, 1.0 - args.density))
    pred = aa > tau if tau > 0 else aa > 0
    yE, yI, ynone = g > 0, g < 0, g == 0
    pE, pI = pred & (a > 0), pred & (a < 0)

    # (A) asym distributions among PREDICTED edges, by truth
    def summ(mask):
        v = asym[mask]
        return dict(n=int(mask.sum()), mean=float(v.mean()) if mask.any() else np.nan,
                    med=float(np.median(v)) if mask.any() else np.nan)
    classes = {"E_true": pE & yE, "E_false": pE & ynone,   # excitatory TP vs FP
               "I_true": pI & yI, "I_false": pI & ynone}
    print("\nasym among predicted edges (mean / median):")
    for k, m in classes.items():
        s = summ(m); print(f"  {k:8s} n={s['n']:>9}  mean={s['mean']:.3f}  med={s['med']:.3f}")

    # store CDFs (subsampled) for the plot
    rng = np.random.default_rng(1)
    cdf = {}
    for k, m in classes.items():
        v = asym[m]
        if len(v) > 200_000:
            v = v[rng.choice(len(v), 200_000, replace=False)]
        cdf[f"asym_{k}"] = np.sort(v).astype(np.float32)

    # (B) FIX: keep predicted edges with asym > thr; sweep thr, re-score excitatory
    thr = np.linspace(0.0, 1.0, args.n_thr)
    E_prec = np.zeros_like(thr); E_rec = np.zeros_like(thr)
    I_prec = np.zeros_like(thr); I_rec = np.zeros_like(thr)
    nE_true, nI_true = max(yE.sum(), 1), max(yI.sum(), 1)
    for t, th in enumerate(thr):
        keepE = pE & (asym > th); keepI = pI & (asym > th)
        tpE = (keepE & yE).sum(); tpI = (keepI & yI).sum()
        E_prec[t] = tpE / max(keepE.sum(), 1); E_rec[t] = tpE / nE_true
        I_prec[t] = tpI / max(keepI.sum(), 1); I_rec[t] = tpI / nI_true

    out = dict(N=N, tag=tag, density=args.density, tau=tau, thr=thr,
               E_prec=E_prec, E_rec=E_rec, I_prec=I_prec, I_rec=I_rec,
               base_E_prec=E_prec[0], base_E_rec=E_rec[0], **cdf)
    out_dir = Path(args.out_dir) if args.out_dir else (BASE / "results" / "r8")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"r8_{tag}.npz", **out)

    with open(out_dir / f"r8_{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["asym_thr", "E_prec", "E_rec", "I_prec", "I_rec"])
        for t in range(len(thr)):
            w.writerow([round(thr[t], 4), round(E_prec[t], 4), round(E_rec[t], 4),
                        round(I_prec[t], 4), round(I_rec[t], 4)])
    print(f"\nbaseline (no filter): E_prec={E_prec[0]:.3f} E_rec={E_rec[0]:.3f}")
    print(f"wrote {out_dir}/r8_{tag}.npz and .csv")


if __name__ == "__main__":
    main()
