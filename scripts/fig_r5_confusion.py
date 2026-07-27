"""
R.5 confusion matrices — COMPUTE stage.

R.5 saved only metrics, not the solved matrices, so this re-solves the cached
moments at the operating point shown in the figure and builds the full 3x3
confusion (true E/none/I x predicted E/none/I) at the density-0.10 threshold.

For each of three recording lengths (smallest / middle / largest checkpoint) and
each method, lambda is taken as the one that MAXIMISES excitatory recall in the
r5_<net>.csv (the same "best lambda" the crossover figure plots). OLS has no
lambda. Only the requested methods are solved (default OLS, Lasso, Lasso+Dale),
so N=12500 is a short GPU job, not the full sweep.

Writes  <out-dir>/r5conf_<net>.npz   (confusion counts + labels)

RUN:
  python scripts/fig_r5_confusion.py --root results --net n1250r4 \
      --csv results/r5/r5_n1250r4.csv --device cpu
  # N=12500 -> slurm/run_r5_confusion.slurm
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
from calcium_ar.solvers.from_moments import (
    ols_from_moments, fista_from_moments, dale_from_moments, strongest_entry_types)
from analyze_run import edge_index
from fig_r5_compute import _torch_helpers
BASE = Path(os.environ.get("CALCIUM_AR_WORKDIR", ROOT))

CLASSES = [1, 0, -1]                 # E, none, I  (sign of the weight)
CLABEL = ["E", "none", "I"]


def best_lam(rows, method, tn, key="E_rec"):
    """lambda maximising `key` for (method, T/N); None for lambda-free OLS."""
    # T/N is stored rounded in the CSV; checkpoints are well separated (factors
    # of ~2), so a 1% + 0.5 tolerance matches the right one without ambiguity.
    tol = 0.01 * tn + 0.5
    cand = [(float(r[key]), float(r["lam"])) for r in rows
            if r["method"] == method and abs(float(r["TN"]) - tn) <= tol
            and r["lam"] != "" and r[key] not in ("", "nan")
            and np.isfinite(float(r[key]))]
    if not cand:
        return None
    return max(cand, key=lambda t: t[0])[1]


def solve_one(method, Cxx, Cyx, lam, l1_ratio, n_fista, n_dale, device, torch):
    l1e, l2e = (lam or 0.0) * l1_ratio, (lam or 0.0) * (1.0 - l1_ratio)
    if device == "cuda":
        _, fista, dale, stypes = _torch_helpers(torch)
        if method == "ols":
            eye = torch.eye(Cxx.shape[0], device=Cxx.device, dtype=Cxx.dtype)
            A = Cyx @ torch.linalg.inv(Cxx + 1e-9 * eye); A.fill_diagonal_(0.0)
            return A
        if method == "lasso":
            return fista(Cxx, Cyx, lam, 0.0, n_fista)
        if method == "en":
            return fista(Cxx, Cyx, l1e, l2e, n_fista)
        if method == "lassodale":
            return dale(Cxx, Cyx, stypes(fista(Cxx, Cyx, lam, 0.0, n_fista)),
                        lam, 0.0, n_dale)
        if method == "endale":
            return dale(Cxx, Cyx, stypes(fista(Cxx, Cyx, l1e, l2e, n_fista)),
                        l1e, l2e, n_dale)
    else:
        if method == "ols":
            return ols_from_moments(Cxx, Cyx)
        if method == "lasso":
            return fista_from_moments(Cxx, Cyx, lam, 0.0, n_fista)
        if method == "en":
            return fista_from_moments(Cxx, Cyx, l1e, l2e, n_fista)
        if method == "lassodale":
            A_l = fista_from_moments(Cxx, Cyx, lam, 0.0, n_fista)
            return dale_from_moments(Cxx, Cyx, strongest_entry_types(A_l), lam, 0.0, n_dale)
        if method == "endale":
            A_e = fista_from_moments(Cxx, Cyx, l1e, l2e, n_fista)
            return dale_from_moments(Cxx, Cyx, strongest_entry_types(A_e), l1e, l2e, n_dale)
    raise ValueError(method)


def confusion(a, g, density):
    """3x3 counts: rows = true class, cols = predicted class (E, none, I)."""
    aa = np.abs(a); y_true = np.sign(g).astype(int)
    tau = float(np.quantile(aa, 1.0 - density))
    conn = aa > tau if tau > 0 else aa > 0
    y_pred = np.where(conn, np.sign(a), 0).astype(int)
    C = np.zeros((3, 3), dtype=np.int64)
    for ti, tc in enumerate(CLASSES):
        for pi, pc in enumerate(CLASSES):
            C[ti, pi] = int(((y_true == tc) & (y_pred == pc)).sum())
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(BASE / "results"))
    ap.add_argument("--net", required=True)
    ap.add_argument("--csv", required=True, help="r5_<net>.csv (for best lambda)")
    ap.add_argument("--methods", nargs="+", default=["ols", "lasso", "lassodale"])
    ap.add_argument("--n-tn", type=int, default=3, help="how many T/N (spread)")
    ap.add_argument("--l1-ratio", type=float, default=0.5)
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

    rows = list(csv.DictReader(open(args.csv)))
    root = Path(args.root); prefix = f"wrapup_{args.net}_T"
    ckpts = sorted(root.glob(f"{prefix}*k"),
                   key=lambda p: int(p.name.rsplit("_T", 1)[1].rstrip("k")))
    if not ckpts:
        raise SystemExit(f"no {prefix}*k under {root}")
    # smallest / middle / largest
    if len(ckpts) >= args.n_tn and args.n_tn == 3:
        pick = [ckpts[0], ckpts[len(ckpts) // 2], ckpts[-1]]
    else:
        pick = ckpts

    out = {"methods": np.array(args.methods), "clabel": np.array(CLABEL)}
    tn_list, conf_list = [], []

    for d in pick:
        sd = d / "seed1"
        if not (sd / "Cxx.npy").exists():
            print(f"[skip] {d.name}: no moments"); continue
        T_ms = float(d.name.rsplit("_T", 1)[1].rstrip("k")) * 1000.0
        Cxx = np.load(sd / "Cxx.npy"); Cyx = np.load(sd / "Cyx.npy")
        adj = np.load(sd / "adj_true.npy", mmap_mode="r"); N = adj.shape[0]
        TN = (T_ms / 0.1) / N
        i, j, _ = edge_index(N, args.sample, np.random.default_rng(0))
        g = np.asarray(adj[j, i], dtype=np.float64); del adj

        if args.device == "cuda":
            Cxx_t = torch.tensor(Cxx, device="cuda", dtype=torch.float32)
            Cyx_t = torch.tensor(Cyx, device="cuda", dtype=torch.float32)
            ii = torch.tensor(i, device="cuda"); jj = torch.tensor(j, device="cuda")
            gather = lambda A: A[ii, jj].detach().float().cpu().numpy().astype(np.float64)
        else:
            Cxx_t, Cyx_t = Cxx, Cyx
            gather = lambda A: A[i, j]

        print(f"\n=== {d.name}  N={N}  T/N={TN:.0f} ===", flush=True)
        mats = []
        for m in args.methods:
            lam = None if m == "ols" else best_lam(rows, m, TN)
            A = solve_one(m, Cxx_t, Cyx_t, lam, args.l1_ratio,
                          args.n_fista, args.n_dale, args.device, torch)
            C = confusion(gather(A), g, args.density)
            mats.append(C); del A
            if args.device == "cuda":
                torch.cuda.empty_cache()
            rec = np.diag(C) / np.maximum(C.sum(1), 1)
            print(f"  {m:10s} lam={('%.2e'%lam) if lam else '  --  '}  "
                  f"recall E/none/I = {rec[0]:.2f}/{rec[1]:.2f}/{rec[2]:.2f}", flush=True)
        tn_list.append(TN); conf_list.append(np.stack(mats))
        del Cxx, Cyx, g

    out["TN"] = np.array(tn_list)
    out["conf"] = np.stack(conf_list)        # (n_tn, n_method, 3, 3)
    out["N"] = int(N)
    out_dir = Path(args.out_dir) if args.out_dir else (BASE / "results" / "r5")
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"r5conf_{args.net}.npz"
    np.savez(fp, **out)
    print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
