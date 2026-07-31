"""
Best-results showcase — COMPUTE stage (cluster).

Ground truth vs inferred connectivity for the best realistic run, reduced to two
small views so nothing large travels home:

  FULL view : the whole N x N matrix block-averaged (signed) to ~600 x 600. Since
              excitatory sources (columns) are positive and inhibitory negative,
              this shows the global E/I band structure — is it recovered?
  SUB-block : a patch chosen to contain BOTH excitatory and inhibitory neurons as
              sources and targets, at full resolution — individual edges.
  ERROR map : the sub-block, each entry classified as TN / TP-E / TP-I / FP / FN /
              wrong-sign against a density-quantile threshold tau fit on a large
              random sample of the FULL matrix (same convention as fig_r1). FP and
              FN are the two error types the showcase figure highlights.

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
    """Signed block-average of M down to about target x target.

    Returns the averaged matrix and the block factor b actually used, so the
    caller can decide whether "block-averaged" is worth telling the viewer
    (b=1 or 2 barely changes the picture; b=20 does).
    """
    N = M.shape[0]
    b = max(1, N // target)
    n = (N // b) * b
    Mb = M[:n, :n].reshape(n // b, b, n // b, b).mean(axis=(1, 3))
    return Mb, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="seed dir (A_ols.npy, adj_true.npy)")
    ap.add_argument("--method", default="ols")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sub-exc", type=int, default=60, help="exc neurons in sub-block")
    ap.add_argument("--sub-inh", type=int, default=40, help="inh neurons in sub-block")
    ap.add_argument("--target", type=int, default=600, help="downsample size for full")
    ap.add_argument("--n-sample", type=int, default=2, help="sample neurons per type")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--density", type=float, default=0.10,
                     help="target predicted density used to set the error-map threshold tau")
    ap.add_argument("--tau-sample", type=int, default=5_000_000,
                     help="random off-diagonal edges sampled (from the FULL matrix) to fit tau")
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

    full_gt, block_factor = block_mean(GT, args.target)
    full_est, _ = block_mean(A, args.target)

    # --- error map: classify each sub-block entry against a global density
    # threshold tau (same quantile-of-|weight| convention as fig_r1_compute) ---- #
    rng_tau = np.random.default_rng(0)
    n_tau = min(args.tau_sample, N * (N - 1))
    ti, tj = rng_tau.integers(0, N, n_tau), rng_tau.integers(0, N, n_tau)
    keep = ti != tj
    tau = float(np.quantile(np.abs(A[ti[keep], tj[keep]]), 1.0 - args.density))

    sub_gt_full = GT[np.ix_(sub, sub)]
    sub_est_full = A[np.ix_(sub, sub)]
    true_sign = np.sign(sub_gt_full)
    est_sign = np.sign(sub_est_full)
    conn = np.abs(sub_est_full) > tau if tau > 0 else np.abs(sub_est_full) > 0

    sub_err = np.zeros(sub_gt_full.shape, dtype=np.int8)          # 0 = TN
    sub_err[(true_sign == 0) & conn] = 3                          # FP
    sub_err[(true_sign != 0) & ~conn] = 4                         # FN
    hit = (true_sign != 0) & conn & (est_sign == true_sign)
    sub_err[hit & (true_sign > 0)] = 1                            # TP-E
    sub_err[hit & (true_sign < 0)] = 2                            # TP-I
    sub_err[(true_sign != 0) & conn & (est_sign != true_sign)] = 5  # wrong-sign

    # sample individual neurons (2 exc + 2 inh by default) and keep their full
    # outgoing profile (all N targets) — columns are the source convention here
    # (see module docstring), so column j of GT/A is neuron j's outgoing row.
    rng = np.random.default_rng(args.sample_seed)
    samp_exc = rng.choice(exc, size=min(args.n_sample, len(exc)), replace=False)
    samp_inh = rng.choice(inh, size=min(args.n_sample, len(inh)), replace=False)
    sample_idx = np.concatenate([samp_exc, samp_inh])
    sample_is_exc = np.concatenate([np.ones(len(samp_exc), dtype=bool),
                                     np.zeros(len(samp_inh), dtype=bool)])

    out = dict(N=N, tag=args.tag, method=args.method, n_exc_sub=n_exc_sub,
               sub_size=len(sub), block_factor=block_factor,
               full_gt=full_gt.astype(np.float32),
               full_est=full_est.astype(np.float32),
               sub_gt=sub_gt_full.astype(np.float32),
               sub_est=sub_est_full.astype(np.float32),
               sub_err=sub_err, tau=tau, err_density=args.density,
               n_exc_full=int((types > 0).sum()),
               sample_idx=sample_idx, sample_is_exc=sample_is_exc,
               sample_est=A[:, sample_idx].T.astype(np.float32))

    out_dir = Path(args.out_dir) if args.out_dir else (BASE / "results" / "best")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"best_{args.tag}.npz", **out)
    n_fp, n_fn = int((sub_err == 3).sum()), int((sub_err == 4).sum())
    print(f"N={N}  sub-block {len(sub)} ({n_exc_sub} exc + {len(sub)-n_exc_sub} inh)  "
          f"tau={tau:.4g}  FP={n_fp}  FN={n_fn}")
    print(f"wrote {out_dir}/best_{args.tag}.npz")


if __name__ == "__main__":
    main()
