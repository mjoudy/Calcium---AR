"""
R.8 (shared input vs directionality) — REGULARIZED-estimator COMPUTE stage.

Same diagnostic as fig_r8_compute.py (asymmetry + false-positive FF/FT split
by reverse-direction truth), but run on Lasso / Lasso+Dale instead of OLS —
testing the professor's hypothesis directly: are R.8's false positives a side
effect of OLS/L2 loss producing a diffuse cloud of small nonzero entries, not
a genuine shared-input confound? If that were the whole story, an
L1-regularized estimator (which drives small entries to EXACTLY zero instead
of leaving noise everywhere) should show a different false-positive
population — fewer of them, and/or without the same symmetric signature.

Solves directly from the cached second moments (Cxx, Cyx) via
calcium_ar.solvers.from_moments — no resimulation, no raw feed, N=1250 only,
~1-2 min on CPU. Uses:
  - the SAME cached connectivity as the Way2/Way3/OU-linear diagnostics
    (results/best_moments/n1250r4/{Cxx,Cyx,adj_true}.npy)
  - the SAME lambda already chosen (max excitatory recall) by the R.5 sweep
    at its longest checkpoint (results/r5/r5_n1250r4.csv, T/N=4000, i.e.
    T=500,000 ms) — not a new hyperparameter search.

NOTE: this uses the T=500k ms snapshot (the only one with cached moments +
a tuned lambda), not the T=20M ms checkpoint used in
fig_R8_violin_fpsplit_n1250 (OLS). N=1250's excitatory recall is flat from
T=500k through T=20M ms (see notebook 2026-08-08), so this is a fair
like-for-like comparison — just worth stating explicitly when reporting.

Writes the SAME npz field layout as fig_r8_compute.py (asym_<class>,
n_<class>, ...) so the existing scripts/fig_r8_violin_fpsplit.py plot script
works unchanged on the output.

Usage:
  python scripts/fig_r8_compute_regularized.py --method lasso
  python scripts/fig_r8_compute_regularized.py --method lassodale
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from analyze_run import edge_index
from calcium_ar.solvers.from_moments import (
    fista_from_moments, dale_from_moments, strongest_entry_types)

EPS = 1e-12
CALCIUM_RESULTS = Path.home() / "calcium_results"
MOMENTS_DIR = CALCIUM_RESULTS / "best_moments" / "n1250r4"
R5_CSV = CALCIUM_RESULTS / "r5" / "r5_n1250r4.csv"
TN = 4000.0  # T/N of the cached moments = 500,000 ms / 1250 neurons


def best_lam(rows, method, tn, key="E_rec"):
    """lambda maximising `key` for (method, T/N) in the R.5 sweep CSV — same
    rule fig_r5_confusion.py uses, so this isn't a fresh hyperparameter pick.
    NOTE: this criterion favors recall, so it tends to pick a barely-
    regularized (near-dense) lambda — see --lam-over-max for a genuinely
    sparse operating point instead."""
    tol = 0.01 * tn + 0.5
    cand = [(float(r[key]), float(r["lam"])) for r in rows
            if r["method"] == method and abs(float(r["TN"]) - tn) <= tol
            and r["lam"] not in ("", "None") and r[key] not in ("", "nan")]
    if not cand:
        raise SystemExit(f"no {method} rows near TN={tn} in {R5_CSV}")
    return max(cand, key=lambda t: t[0])[1]


def lam_at_ratio(rows, method, tn, ratio, tol_ratio=0.05):
    """lambda closest to a given lam/lam_max ratio — for a genuinely sparse
    operating point (best_lam's E_rec criterion picks a near-dense one)."""
    tol = 0.01 * tn + 0.5
    cand = [(abs(float(r["lam_over_max"]) - ratio), float(r["lam"]), float(r["E_rec"]))
            for r in rows if r["method"] == method and abs(float(r["TN"]) - tn) <= tol
            and r["lam"] not in ("", "None")]
    if not cand:
        raise SystemExit(f"no {method} rows near TN={tn} in {R5_CSV}")
    cand.sort(key=lambda t: t[0])
    return cand[0][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["lasso", "lassodale"], required=True)
    ap.add_argument("--strength", choices=["weak", "strong"], default="weak",
                    help="weak = best_lam(E_rec), near-dense; "
                         "strong = lam_over_max~0.25, genuinely sparse")
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--sample", type=int, default=20_000_000)
    ap.add_argument("--n-fista", type=int, default=500)
    ap.add_argument("--n-dale", type=int, default=800)
    ap.add_argument("--out-dir", default=str(CALCIUM_RESULTS / "r8"))
    args = ap.parse_args()

    print(f"reading moments from {MOMENTS_DIR}")
    Cxx = np.load(MOMENTS_DIR / "Cxx.npy")
    Cyx = np.load(MOMENTS_DIR / "Cyx.npy")
    adj = np.load(MOMENTS_DIR / "adj_true.npy", mmap_mode="r")
    N = Cxx.shape[0]

    rows = list(csv.DictReader(open(R5_CSV)))
    if args.strength == "weak":
        lam = best_lam(rows, args.method, TN)
        print(f"method={args.method}  strength=weak  lam={lam:.4e}  "
              f"(best E_rec @ T/N={TN:.0f}, from {R5_CSV.name})")
    else:
        lam = lam_at_ratio(rows, args.method, TN, ratio=0.25)
        print(f"method={args.method}  strength=strong  lam={lam:.4e}  "
              f"(lam/lam_max~0.25 @ T/N={TN:.0f}, from {R5_CSV.name})")

    A_lasso = fista_from_moments(Cxx, Cyx, lam, 0.0, args.n_fista)
    if args.method == "lasso":
        A = A_lasso
    else:
        types = strongest_entry_types(A_lasso)
        A = dale_from_moments(Cxx, Cyx, types, lam, 0.0, args.n_dale)
    nz = int((A != 0).sum())
    print(f"solved: nnz={nz:,} / {N*N:,} ({nz/(N*N):.1%} dense)")

    i, j, sampled = edge_index(N, args.sample, np.random.default_rng(0))
    a = np.asarray(A[i, j], dtype=np.float64)       # inferred j->i
    a_rev = np.asarray(A[j, i], dtype=np.float64)   # inferred i->j
    g = np.asarray(adj[j, i], dtype=np.float64)      # true j->i
    g_rev = np.asarray(adj[i, j], dtype=np.float64)  # true i->j (reverse)
    aa = np.abs(a)
    asym = np.abs(a - a_rev) / (np.abs(a) + np.abs(a_rev) + EPS)

    tau = float(np.quantile(aa, 1.0 - args.density))
    pred = aa > tau if tau > 0 else aa > 0
    yE, yI, ynone = g > 0, g < 0, g == 0
    y_rev_conn, y_rev_none = g_rev != 0, g_rev == 0
    pE, pI = pred & (a > 0), pred & (a < 0)

    classes = {"E_true": pE & yE, "E_false": pE & ynone,
               "E_false_FF": pE & ynone & y_rev_none,
               "E_false_FT": pE & ynone & y_rev_conn,
               "I_true": pI & yI, "I_false": pI & ynone,
               "I_false_FF": pI & ynone & y_rev_none,
               "I_false_FT": pI & ynone & y_rev_conn}

    print(f"\nasym among predicted edges ({args.method}, mean / median):")
    cdf = {}
    rng = np.random.default_rng(1)
    for k, m in classes.items():
        v = asym[m]
        n = int(m.sum())
        mean = float(v.mean()) if n else float("nan")
        med = float(np.median(v)) if n else float("nan")
        print(f"  {k:12s} n={n:>9,}  mean={mean:.3f}  med={med:.3f}")
        cdf[f"n_{k}"] = n
        vv = v
        if len(vv) > 200_000:
            keep = rng.choice(len(vv), 200_000, replace=False)
            vv = vv[keep]
        cdf[f"asym_{k}"] = np.sort(vv).astype(np.float32)

    out = dict(N=N, tag=f"n1250r4_{args.method}_{args.strength}", density=args.density,
               tau=tau, method=args.method, strength=args.strength, lam=lam, TN=TN, **cdf)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"r8_wrapup_n1250r4_{args.method}_{args.strength}.npz"
    np.savez(fp, **out)
    print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
