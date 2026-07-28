"""
R.7 (hidden neurons / shared input) — COMPUTE stage.

Real recordings see only a fraction of the network; the unobserved neurons feed
SHARED INPUT into the observed ones and create spurious edges. This is the
mechanism behind the confounding-limited excitatory ceiling seen in R.1-R.6.

Partial observation is a re-solve of CACHED moments — no new simulation. Observing
a subset S means solving OLS on the sub-blocks Cxx[S,S], Cyx[S,S]: a pairwise
moment <x_i x_j> does not depend on who else was recorded, so the sub-block IS the
observed-subset moment matrix. The confounding enters through inv(Cxx[S,S]), which
cannot subtract out the hidden common input.

Two outputs:

1. DEGRADATION curve. Observation fraction f = 1.0 / 0.5 / 0.25 / 0.1, with the
   recording length chosen per fraction so samples-per-observed-neuron T/|S| stays
   ~constant (uses the R.5 short-T checkpoints). This ISOLATES hidden-input
   confounding from the data-per-neuron effect (which would otherwise rise as S
   shrinks). Nested subsets (S_0.1 c S_0.25 c ...), E:I ratio preserved.

2. MECHANISM panel. At one fraction, for observed pairs that are NOT truly
   connected, the inferred |weight| binned by the number of COMMON presynaptic
   neurons (over the full network, hidden included). If shared input causes false
   edges, |weight| rises with the overlap.

Writes  <out-dir>/r7_<net>.npz  and  r7_<net>.csv

RUN: see slurm/run_r7.slurm  (N=12500 wants the GPU for the inverses).
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
from analyze_run import edge_index, score
BASE = Path(os.environ.get("CALCIUM_AR_WORKDIR", ROOT))
DT = 0.1


def infer_types(adj):
    """+1 excitatory / -1 inhibitory from each neuron's outgoing (row) sign."""
    s = np.sign(adj.sum(1))
    s[s == 0] = 1.0
    return s


def nested_subsets(types, fractions, seed=0):
    """{f: index array}, stratified by E/I and NESTED (prefixes of one shuffle)."""
    rng = np.random.default_rng(seed)
    E = np.flatnonzero(types > 0); I = np.flatnonzero(types < 0)
    rng.shuffle(E); rng.shuffle(I)
    out = {}
    for f in fractions:
        nE, nI = int(round(f * len(E))), int(round(f * len(I)))
        out[f] = np.sort(np.concatenate([E[:nE], I[:nI]]))
    return out


def solve_ols(Cxx, Cyx, device, torch):
    if device == "cuda":
        N = Cxx.shape[0]
        A = Cyx @ torch.linalg.inv(Cxx + 1e-9 * torch.eye(N, device=Cxx.device,
                                                           dtype=Cxx.dtype))
        A.fill_diagonal_(0.0)
        return A
    A = Cyx @ np.linalg.inv(Cxx + 1e-9 * np.eye(Cxx.shape[0]))
    np.fill_diagonal(A, 0.0)
    return A


def shared_panel(adj, S, A_SS, torch, device, max_pairs=40_000_000, seed=0):
    """Mean inferred |weight| of NON-connected observed pairs, binned by their
    number of common presynaptic neurons over the FULL network."""
    # sources into target t = column t of adj (adj[source, target]); shared count
    # of targets (i, j) = sum_s 1[adj[s,i]] 1[adj[s,j]] = (Bc[:,S]^T Bc[:,S]).
    Bfull = (adj != 0).astype(np.float32)          # (source, target)
    if device == "cuda":
        Bt = torch.tensor(Bfull[:, S], device="cuda")       # N x |S|
        shared = (Bt.T @ Bt).cpu().numpy()                  # |S| x |S|
        A = A_SS.detach().float().cpu().numpy()
    else:
        Bs = Bfull[:, S]
        shared = Bs.T @ Bs
        A = A_SS
    adj_SS = adj[np.ix_(S, S)]
    ns = len(S)
    iu, ju = np.triu_indices(ns, k=1)              # unordered observed pairs
    # truly UNconnected pair: no direct edge either way
    noedge = (adj_SS[iu, ju] == 0) & (adj_SS[ju, iu] == 0)
    iu, ju = iu[noedge], ju[noedge]
    if len(iu) > max_pairs:
        r = np.random.default_rng(seed).choice(len(iu), max_pairs, replace=False)
        iu, ju = iu[r], ju[r]
    sh = shared[iu, ju].astype(int)
    w = np.maximum(np.abs(A[iu, ju]), np.abs(A[ju, iu]))   # spurious magnitude
    bins = np.arange(0, sh.max() + 2)
    mean_w = np.full(len(bins) - 1, np.nan); cnt = np.zeros(len(bins) - 1)
    for b in range(len(bins) - 1):
        m = sh == bins[b]
        cnt[b] = m.sum()
        if m.any():
            mean_w[b] = w[m].mean()
    return bins[:-1], mean_w, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(BASE / "results"))
    ap.add_argument("--net", default="n12500r4")
    ap.add_argument("--fractions", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.1])
    ap.add_argument("--base-tn", type=float, default=800.0,
                    help="target samples per observed neuron T/|S|")
    ap.add_argument("--shared-frac", type=float, default=0.5)
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--sample", type=int, default=20_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    torch = None
    if args.device == "cuda":
        import torch as _t; torch = _t
        assert torch.cuda.is_available(), "no CUDA — use --device cpu"

    root = Path(args.root); prefix = f"wrapup_{args.net}_T"
    ckpts = {}
    for d in root.glob(f"{prefix}*k"):
        sd = d / "seed1"
        if (sd / "Cxx.npy").exists():
            T_ms = float(d.name.rsplit("_T", 1)[1].rstrip("k")) * 1000.0
            ckpts[T_ms] = sd
    if not ckpts:
        raise SystemExit(f"no checkpoints with moments under {root} ({prefix}*)")
    Tavail = np.array(sorted(ckpts))

    # types/subsets from any checkpoint's adj_true (identical across lengths)
    adj0 = np.load(ckpts[Tavail[0]] / "adj_true.npy")
    N = adj0.shape[0]; types = infer_types(adj0)
    subs = nested_subsets(types, args.fractions, args.seed)
    print(f"N={N}  NE={(types>0).sum()}  NI={(types<0).sum()}  "
          f"checkpoints T(ms)={list(Tavail.astype(int))}")

    rows, curve = [], []
    shared_out = None
    for f in args.fractions:
        S = subs[f]; nS = len(S)
        T_want = args.base_tn * nS * DT                 # ms for target T/|S|
        T_use = float(Tavail[np.argmin(np.abs(Tavail - T_want))])
        sd = ckpts[T_use]
        adj = np.load(sd / "adj_true.npy", mmap_mode="r")
        Cxx = np.load(sd / "Cxx.npy"); Cyx = np.load(sd / "Cyx.npy")
        ix = np.ix_(S, S)
        Cxx_s = np.ascontiguousarray(Cxx[ix]); Cyx_s = np.ascontiguousarray(Cyx[ix])
        del Cxx, Cyx
        if args.device == "cuda":
            Cxx_t = torch.tensor(Cxx_s, device="cuda", dtype=torch.float32)
            Cyx_t = torch.tensor(Cyx_s, device="cuda", dtype=torch.float32)
            A = solve_ols(Cxx_t, Cyx_t, "cuda", torch)
            A_np = A.detach().float().cpu().numpy().astype(np.float64)
        else:
            A = solve_ols(Cxx_s, Cyx_s, "cpu", torch)
            A_np = A

        adj_SS = np.asarray(adj[np.ix_(S, S)], dtype=np.float64)
        i, j, sampled = edge_index(nS, args.sample, np.random.default_rng(0))
        g = adj_SS.T[i, j]; a = A_np[i, j]
        r = score(a, g, args.density)
        TN = (T_use / DT) / nS
        r.update(net=args.net, frac=f, n_obs=nS, T_ms=int(T_use), TN=round(TN, 1))
        rows.append(r); curve.append((f, nS, TN, r))
        print(f"  f={f:>4}  |S|={nS:>6}  T={int(T_use)//1000}k  T/|S|={TN:>6.0f}  "
              f"corr={r['corr']:.3f}  E_rec={r['E_rec']:.3f}  E_prec={r['E_prec']:.3f}",
              flush=True)

        if abs(f - args.shared_frac) < 1e-9:
            A_for_panel = A if args.device == "cuda" else A_np
            sb, sw, sc = shared_panel(np.asarray(adj), S, A_for_panel, torch,
                                      args.device, seed=args.seed)
            shared_out = dict(shared_bins=sb, shared_meanw=sw, shared_count=sc,
                              shared_frac=f)
            print(f"  [shared panel @ f={f}] bins 0..{sb[-1]}, "
                  f"non-edge pairs binned", flush=True)
        del adj, Cxx_s, Cyx_s, A_np
        if args.device == "cuda":
            del A; torch.cuda.empty_cache()

    out_dir = Path(args.out_dir) if args.out_dir else (BASE / "results" / "r7")
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = dict(N=N, fractions=np.array([c[0] for c in curve]),
               n_obs=np.array([c[1] for c in curve]),
               TN=np.array([c[2] for c in curve]),
               corr=np.array([c[3]["corr"] for c in curve]),
               E_rec=np.array([c[3]["E_rec"] for c in curve]),
               E_prec=np.array([c[3]["E_prec"] for c in curve]),
               I_rec=np.array([c[3]["I_rec"] for c in curve]),
               I_prec=np.array([c[3]["I_prec"] for c in curve]),
               auc_E=np.array([c[3]["auc_E"] for c in curve]))
    if shared_out:
        npz.update(shared_out)
    np.savez(out_dir / f"r7_{args.net}.npz", **npz)

    fields = ["net", "frac", "n_obs", "T_ms", "TN", "corr", "roc_auc", "pr_ap",
              "auc_E", "auc_I", "E_rec", "E_prec", "I_rec", "I_prec", "none_rec"]
    with open(out_dir / f"r7_{args.net}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nwrote {out_dir}/r7_{args.net}.npz and .csv")


if __name__ == "__main__":
    main()
