"""
R.5 (regularization vs data) — COMPUTE stage.

Question: when data is scarce, does regularization cure it? OLS is unbiased but
high-variance at short recordings; Ridge/Lasso/EN trade a little bias for less
variance. So there should be a CROSSOVER in T: regularization wins when T/N is
small, OLS catches up once T/N is large. And a second, structural limit sits
underneath — confounding bias, which regularization CANNOT remove — so the cure
is expected only at the data-scarce end.

This stage does NO simulation. It re-solves the cached moment matrices (Cxx, Cyx)
that the R.4 runs already saved with --save-moments, for every estimator across a
lambda grid, and scores each with the SAME scorer as R.4 (scripts/analyze_run.py)
so the numbers are directly comparable.

The lambda grid is DATA-DERIVED, not hardcoded: lam_max = max|Cyx off-diagonal|
is the smallest L1 penalty that zeros every weight; we sweep from lam_max down to
lam_max/1000. This auto-adapts to each size (J differs), so no hand-scaling.

Reads   <root>/<prefix>_T*k/seed1/{Cxx,Cyx,adj_true}.npy
Writes  <out-dir>/r5_<net>.csv   (one row per checkpoint x method x lambda)

RUN:
  # N=1250 (small, local or cpu):
  python scripts/fig_r5_compute.py --root results --net n1250r4 --device cpu
  # N=12500 (large, GPU): see slurm/run_r5.slurm
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from calcium_ar.solvers.from_moments import (
    ols_from_moments, fista_from_moments, dale_from_moments, strongest_entry_types)
from analyze_run import edge_index, score, FIELDS as SCORE_FIELDS

METHODS = ["ols", "ridge", "lasso", "en", "lassodale", "endale"]
OUT_FIELDS = ["net", "N", "T_ms", "TN", "method", "lam", "lam_over_max"] + \
             [f for f in SCORE_FIELDS if f not in ("run", "seed", "method")]


# --------------------------------------------------------------------------- #
# torch solvers (GPU) — exact same iteration as the numpy from_moments ones    #
# --------------------------------------------------------------------------- #
def _torch_helpers(torch):
    def pow_eig(C, iters=60):
        v = torch.randn(C.shape[0], device=C.device, dtype=C.dtype)
        v /= v.norm()
        for _ in range(iters):
            v = C @ v; v /= v.norm()
        return float(v @ (C @ v))

    def fista(Cxx, Cyx, l1, l2, n_iter):
        L = pow_eig(Cxx) + l2; step = 1.0 / L
        A = torch.zeros_like(Cxx); Ap = A.clone(); tk = 1.0
        for _ in range(n_iter):
            tn = 0.5 * (1.0 + (1.0 + 4.0 * tk * tk) ** 0.5)
            y = A + ((tk - 1.0) / tn) * (A - Ap)
            grad = y @ Cxx - Cyx + l2 * y
            Ap = A
            u = y - step * grad
            A = torch.sign(u) * torch.clamp(u.abs() - step * l1, min=0.0)
            tk = tn
        A.fill_diagonal_(0.0); return A

    def dale(Cxx, Cyx, types, l1, l2, n_iter):
        L = pow_eig(Cxx) + l2; thr = l1 / L
        tcol = types.view(1, -1)
        A = torch.zeros_like(Cxx); Z = A.clone(); tk = 1.0
        for _ in range(n_iter):
            V = Z - (Z @ Cxx - Cyx + l2 * Z) / L
            V = torch.sign(V) * torch.clamp(V.abs() - thr, min=0.0)
            V = tcol * torch.clamp(tcol * V, min=0.0)
            tn = (1.0 + (1.0 + 4.0 * tk * tk) ** 0.5) / 2.0
            Z = V + ((tk - 1.0) / tn) * (V - A); A = V; tk = tn
        A.fill_diagonal_(0.0); return A

    def strongest_types(A):
        Aa = A.abs().clone(); Aa.fill_diagonal_(0.0)
        jmax = Aa.argmax(dim=0)
        cols = torch.arange(A.shape[0], device=A.device)
        s = torch.sign(A[jmax, cols])
        s[s == 0] = 1.0
        return s

    return pow_eig, fista, dale, strongest_types


def solve_all(Cxx, Cyx, lam, l1_ratio, n_fista, n_dale, device, torch=None):
    """Return {method: A_numpy_or_gpu} for one lambda. On cuda the matrices stay
    on the GPU; the caller gathers sampled entries there."""
    out = {}
    l1e, l2e = lam * l1_ratio, lam * (1.0 - l1_ratio)   # elastic-net split
    if device == "cuda":
        _, fista, dale, stypes = _torch_helpers(torch)
        eye = torch.eye(Cxx.shape[0], device=Cxx.device, dtype=Cxx.dtype)
        out["ridge"] = Cyx @ torch.linalg.inv(Cxx + lam * eye)
        out["ridge"].fill_diagonal_(0.0)
        out["lasso"] = fista(Cxx, Cyx, lam, 0.0, n_fista)
        out["en"] = fista(Cxx, Cyx, l1e, l2e, n_fista)
        out["lassodale"] = dale(Cxx, Cyx, stypes(out["lasso"]), lam, 0.0, n_dale)
        out["endale"] = dale(Cxx, Cyx, stypes(out["en"]), l1e, l2e, n_dale)
    else:
        out["ridge"] = ols_from_moments(Cxx, Cyx, ridge=lam)
        out["lasso"] = fista_from_moments(Cxx, Cyx, lam, 0.0, n_fista)
        out["en"] = fista_from_moments(Cxx, Cyx, l1e, l2e, n_fista)
        out["lassodale"] = dale_from_moments(
            Cxx, Cyx, strongest_entry_types(out["lasso"]), lam, 0.0, n_dale)
        out["endale"] = dale_from_moments(
            Cxx, Cyx, strongest_entry_types(out["en"]), l1e, l2e, n_dale)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results", help="dir holding <prefix>_T*k/")
    ap.add_argument("--net", required=True, help="preset stem, e.g. n1250r4")
    ap.add_argument("--n-lam", type=int, default=6)
    ap.add_argument("--lam-lo", type=float, default=1e-3,
                    help="grid floor as a fraction of lam_max")
    ap.add_argument("--l1-ratio", type=float, default=0.5, help="EN L1 fraction")
    ap.add_argument("--n-fista", type=int, default=500)
    ap.add_argument("--n-dale", type=int, default=800)
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--sample", type=int, default=20_000_000)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    torch = None
    if args.device == "cuda":
        import torch as _t; torch = _t
        assert torch.cuda.is_available(), "no CUDA — use --device cpu"
        print("gpu:", torch.cuda.get_device_name(0))

    root = Path(args.root)
    prefix = f"wrapup_{args.net}_T"
    ckpts = sorted(root.glob(f"{prefix}*k"),
                   key=lambda p: int(p.name.rsplit("_T", 1)[1].rstrip("k")))
    if not ckpts:
        raise SystemExit(f"no {prefix}*k dirs under {root}")

    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "results" / "r5")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for d in ckpts:
        sd = d / "seed1"
        need = [sd / "Cxx.npy", sd / "Cyx.npy", sd / "adj_true.npy"]
        if not all(p.exists() for p in need):
            print(f"[skip] {d.name}: missing moments/adj (re-stream with "
                  f"--save-moments)"); continue
        T_ms = float(d.name.rsplit("_T", 1)[1].rstrip("k")) * 1000.0
        Cxx = np.load(need[0]); Cyx = np.load(need[1])
        adj = np.load(need[2], mmap_mode="r")
        N = adj.shape[0]; TN = (T_ms / 0.1) / N

        # sampled edges: SAME set for every method/lambda at this checkpoint
        rng = np.random.default_rng(0)
        i, j, sampled = edge_index(N, args.sample, rng)
        g = np.asarray(adj[j, i], dtype=np.float64)     # A[i,j] <-> adj.T[i,j]
        del adj

        lam_max = float(np.abs(Cyx[~np.eye(N, dtype=bool)]).max())
        grid = np.geomspace(lam_max * args.lam_lo, lam_max, args.n_lam)
        print(f"\n=== {d.name}  N={N}  T/N={TN:.0f}  lam_max={lam_max:.3e} ===",
              flush=True)

        if args.device == "cuda":
            Cxx_t = torch.tensor(Cxx, device="cuda", dtype=torch.float32)
            Cyx_t = torch.tensor(Cyx, device="cuda", dtype=torch.float32)
            ii = torch.tensor(i, device="cuda"); jj = torch.tensor(j, device="cuda")
            gather = lambda A: A[ii, jj].detach().float().cpu().numpy().astype(np.float64)
            A_ols = Cyx_t @ torch.linalg.inv(
                Cxx_t + 1e-9 * torch.eye(N, device="cuda", dtype=torch.float32))
            A_ols.fill_diagonal_(0.0)
            a_ols = gather(A_ols); del A_ols
        else:
            Cxx_t, Cyx_t = Cxx, Cyx
            gather = lambda A: A[i, j]
            a_ols = gather(ols_from_moments(Cxx, Cyx))

        def record(method, lam, a):
            r = score(a, g, args.density)
            r.update(net=args.net, N=N, T_ms=int(T_ms), TN=round(TN, 1),
                     method=method, lam=(f"{lam:.4e}" if lam is not None else ""),
                     lam_over_max=("" if lam is None else round(lam / lam_max, 5)))
            rows.append(r)
            print(f"  {method:10s} lam={('%.2e'%lam) if lam is not None else '  --  '}"
                  f"  corr={r['corr']:.3f}  AUC_E={r['auc_E']:.3f}  "
                  f"E_rec={r['E_rec']:.3f}", flush=True)

        record("ols", None, a_ols)                      # lambda-independent baseline
        for lam in grid:
            mats = solve_all(Cxx_t, Cyx_t, float(lam), args.l1_ratio,
                             args.n_fista, args.n_dale, args.device, torch)
            for m in ["ridge", "lasso", "en", "lassodale", "endale"]:
                record(m, float(lam), gather(mats[m]))
            del mats
            if args.device == "cuda":
                torch.cuda.empty_cache()
        del Cxx, Cyx, g

    out_path = out_dir / f"r5_{args.net}.csv"
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in OUT_FIELDS})
    print(f"\nwrote {out_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
