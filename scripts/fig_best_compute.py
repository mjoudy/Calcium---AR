"""
Best-results showcase — COMPUTE stage (cluster).

Ground truth vs inferred connectivity for the best realistic run, reduced to two
small views so nothing large travels home:

  FULL view : the whole N x N matrix block-averaged (signed) to ~600 x 600. Since
              excitatory sources (columns) are positive and inhibitory negative,
              this shows the global E/I band structure — is it recovered?
  SUB-block : a patch chosen to contain BOTH excitatory and inhibitory neurons as
              sources and targets, at full resolution — individual edges.

Both are shown in the estimator's orientation (A[i,j] ~ adj.T[i,j]).

Writes <out-dir>/best_<tag>.npz

RUN:  python scripts/fig_best_compute.py --data <seed dir> --tag n12500lr
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = Path(os.environ.get("CALCIUM_AR_WORKDIR", ROOT))


def block_mean(M, target=600):
    """Signed block-average of M down to about target x target."""
    N = M.shape[0]
    b = max(1, N // target)
    n = (N // b) * b
    Mb = M[:n, :n].reshape(n // b, b, n // b, b).mean(axis=(1, 3))
    return Mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="seed dir (A_ols.npy, adj_true.npy)")
    ap.add_argument("--method", default="ols")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sub-exc", type=int, default=60, help="exc neurons in sub-block")
    ap.add_argument("--sub-inh", type=int, default=40, help="inh neurons in sub-block")
    ap.add_argument("--target", type=int, default=600, help="downsample size for full")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    sd = Path(args.data)
    A = np.load(sd / f"A_{args.method}.npy").astype(np.float64)
    adj = np.load(sd / "adj_true.npy").astype(np.float64)
    N = A.shape[0]
    np.fill_diagonal(A, 0.0); np.fill_diagonal(adj, 0.0)
    GT = adj.T                                   # estimator orientation

    # neuron types from outgoing sign; Brunel = exc first, inh last
    types = np.sign(adj.sum(1)); types[types == 0] = 1
    exc = np.flatnonzero(types > 0); inh = np.flatnonzero(types < 0)
    sub = np.concatenate([exc[:args.sub_exc], inh[:args.sub_inh]])
    n_exc_sub = min(args.sub_exc, len(exc))

    out = dict(N=N, tag=args.tag, method=args.method, n_exc_sub=n_exc_sub,
               sub_size=len(sub),
               full_gt=block_mean(GT, args.target).astype(np.float32),
               full_est=block_mean(A, args.target).astype(np.float32),
               sub_gt=GT[np.ix_(sub, sub)].astype(np.float32),
               sub_est=A[np.ix_(sub, sub)].astype(np.float32),
               n_exc_full=int((types > 0).sum()))

    out_dir = Path(args.out_dir) if args.out_dir else (BASE / "results" / "best")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"best_{args.tag}.npz", **out)
    print(f"N={N}  sub-block {len(sub)} ({n_exc_sub} exc + {len(sub)-n_exc_sub} inh)")
    print(f"wrote {out_dir}/best_{args.tag}.npz")


if __name__ == "__main__":
    main()
