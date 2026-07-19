"""
Re-solve estimators from CACHED moments — seconds instead of hours.

Every estimator (OLS, Ridge, EN, Lasso, Dale) is a function of the two N x N
lag-pair moment matrices (Cxx, Cyx). If a run was launched with --save-moments,
those matrices sit next to the estimates, so trying a new lambda, a new Dale
variant, or a new estimator entirely costs one matrix solve — no NEST
simulation, no calcium generation, no deconvolution.

Cost comparison at N=12500, 5M ms:
    full pipeline   ~10 h
    from moments    ~seconds (OLS/Ridge) to ~minutes (FISTA)

Usage:
  # sweep L1 strength on an existing run
  python scripts/solve_from_cached.py --data results/wrapup_n12500_T5000k \
      --methods en --lam-l1 1e-6 1e-5 1e-4 --score

  # add ridge at a different strength
  python scripts/solve_from_cached.py --data results/wrapup_n12500_T5000k \
      --methods ridge --lam-l2 1e-3 --score
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calcium_ar.solvers.from_moments import (
    ols_from_moments, fista_from_moments, dale_from_moments, strongest_entry_types)


def score(A, adj, density=0.10):
    """Lean metrics (large N safe): AUC, per-class AUC, recalls."""
    from sklearn.metrics import roc_auc_score
    N = A.shape[0]
    m = ~np.eye(N, dtype=bool)
    g = adj.T[m].ravel().astype(np.float64)
    a = A[m].ravel().astype(np.float64)
    aa = np.abs(a)
    yE, yI, yc = g > 0, g < 0, g != 0
    tau = float(np.quantile(aa, 1.0 - density))
    conn = aa > tau if tau > 0 else aa > 0
    return dict(
        roc_auc=float(roc_auc_score(yc, aa)),
        auc_E=float(roc_auc_score(yE, a)),
        auc_I=float(roc_auc_score(yI, -a)),
        E_rec=float((conn & (a > 0) & yE).sum() / max(yE.sum(), 1)),
        I_rec=float((conn & (a < 0) & yI).sum() / max(yI.sum(), 1)),
        corr=float(np.corrcoef(a, g)[0, 1]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="run dir containing seed*/Cxx.npy (from --save-moments)")
    ap.add_argument("--seed", default="seed1")
    ap.add_argument("--methods", nargs="+", default=["ols"],
                    choices=["ols", "ridge", "en", "lasso", "endale", "lassodale"])
    ap.add_argument("--lam-l1", type=float, nargs="+", default=[1e-4])
    ap.add_argument("--lam-l2", type=float, nargs="+", default=[1e-4])
    ap.add_argument("--n-iter", type=int, default=500)
    ap.add_argument("--score", action="store_true", help="score against adj_true.npy")
    ap.add_argument("--save", action="store_true", help="write the estimate as .npy")
    args = ap.parse_args()

    d = Path(args.data) / args.seed
    cxx, cyx = d / "Cxx.npy", d / "Cyx.npy"
    if not cxx.exists():
        raise SystemExit(
            f"no cached moments in {d}.\nRe-run the sweep with --save-moments "
            f"(they are not recoverable from the saved estimates).")
    Cxx = np.load(cxx)
    Cyx = np.load(cyx)
    adj = np.load(d / "adj_true.npy").astype(np.float64) if args.score else None
    print(f"loaded moments {Cxx.shape} from {d}")

    for m in args.methods:
        lam1s = args.lam_l1 if m in ("en", "lasso", "endale", "lassodale") else [None]
        lam2s = args.lam_l2 if m in ("ridge", "en", "endale") else [None]
        for l1 in lam1s:
            for l2 in lam2s:
                if m == "ols":
                    A, tag = ols_from_moments(Cxx, Cyx), "ols"
                elif m == "ridge":
                    A, tag = ols_from_moments(Cxx, Cyx, ridge=l2), f"ridge_l2{l2:g}"
                elif m == "en":
                    A, tag = fista_from_moments(Cxx, Cyx, l1, l2, args.n_iter), \
                             f"en_l1{l1:g}_l2{l2:g}"
                elif m == "lasso":
                    A, tag = fista_from_moments(Cxx, Cyx, l1, 0.0, args.n_iter), \
                             f"lasso_l1{l1:g}"
                elif m == "endale":
                    base = fista_from_moments(Cxx, Cyx, l1, l2, args.n_iter)
                    A = dale_from_moments(Cxx, Cyx, strongest_entry_types(base),
                                          l1, l2, args.n_iter + 300)
                    tag = f"endale_l1{l1:g}_l2{l2:g}"
                else:  # lassodale
                    base = fista_from_moments(Cxx, Cyx, l1, 0.0, args.n_iter)
                    A = dale_from_moments(Cxx, Cyx, strongest_entry_types(base),
                                          l1, 0.0, args.n_iter + 300)
                    tag = f"lassodale_l1{l1:g}"

                line = f"{tag:26s} nnz={float((A != 0).mean()):.3f}"
                if args.score:
                    s = score(A, adj)
                    line += ("  AUC={roc_auc:.3f}  AUC_E={auc_E:.3f}  AUC_I={auc_I:.3f}"
                             "  E_rec={E_rec:.3f}  I_rec={I_rec:.3f}  corr={corr:.3f}"
                             ).format(**s)
                print(line, flush=True)
                if args.save:
                    np.save(d / f"A_{tag}.npy", A.astype(np.float32))


if __name__ == "__main__":
    main()
