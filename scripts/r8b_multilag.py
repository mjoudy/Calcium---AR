"""
R.8 phase 2 (multiple lags) — the TIMING axis, measured on its own.

Symmetry (phase 1) asks "is the coupling one-directional?" — a LEFT/RIGHT feature
of the cross-correlation. Timing asks a DIFFERENT question: "does the coupling
peak instantaneously (shared common input) or at the synaptic delay (a real
edge)?" — a CENTRE/DISPLACED feature. The two are independent (see R.8 discussion:
a reciprocal real pair is symmetric BUT delayed).

This reads the cached spikes (no new NEST), streams the multi-lag moments
C(l)=<x(t+l) x(t)^T> for a grid of lags, forms the per-lag connectivity
A_l = C(l) C(0)^{-1}, and for sampled pairs records:
  - the lag PROFILE |A_l[i,j]| and its peak lag  (timing axis)
  - the directional asymmetry at the estimator lag (symmetry axis, phase 1)
  - the true label,
so the two axes can be analysed separately and crossed (2x2).

Saves a small npz (class-averaged profiles + a per-pair subsample) — the big C(l)
matrices never leave the GPU. Plot with fig_r8b_plot.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from calcium_ar.experiments.streaming import stream_moments
from analyze_run import edge_index
from wrapup_run import build_cfg
BASE = Path(os.environ.get("CALCIUM_AR_WORKDIR", ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, help="preset key, e.g. n12500_r4")
    ap.add_argument("--cache-dir", default=str(BASE / "gt_cache"))
    ap.add_argument("--checkpoint-ms", type=float, default=5_000_000.0)
    ap.add_argument("--lags-ms", type=float, nargs="+",
                    default=[0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0])
    ap.add_argument("--signal", default="feed", choices=["feed", "calcium", "spikes"],
                    help="what to build moments on: deconvolved feed / raw calcium / raw spikes")
    ap.add_argument("--chunk-ms", type=float, default=5000.0)
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--sample", type=int, default=20_000_000)
    ap.add_argument("--keep", type=int, default=2_000_000, help="per-pair rows to save")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    torch = None
    if args.device == "cuda":
        import torch
        assert torch.cuda.is_available(), "no CUDA — use --device cpu"

    cfg = build_cfg(args.net)
    dt = cfg["dt"]; N = cfg["n_excitatory"] + cfg["n_inhibitory"]
    smooth_win = max(5, round(cfg["smooth_window_ms"] / dt))
    lags = sorted({max(0, round(l / dt)) for l in args.lags_ms})
    lag_est = max(1, round(cfg["lag_ms"] / dt))
    if lag_est not in lags:
        lags = sorted(set(lags) | {lag_est})
    pos_lags = [l for l in lags if l > 0]
    lags_ms = np.array([l * dt for l in pos_lags])
    print(f"net={args.net} N={N} dt={dt} lag_est={lag_est} ({cfg['lag_ms']}ms)")
    print(f"lags(samples)={lags}  positive={pos_lags}")

    # cached spikes + ground truth: pick the LONGEST recording (tag ..._T<ms/1000>k),
    # not the alphabetically-last file (which mis-picks e.g. T50k over T500k).
    cache = Path(args.cache_dir)
    cand = list(cache.glob(f"{args.net}_seed1_*.npz"))
    if not cand:
        raise SystemExit(f"no cached spikes {args.net}_seed1_*.npz under {cache}")
    def _tk(p):
        import re
        m = re.search(r"_T(\d+)k_", p.name)
        return int(m.group(1)) if m else -1
    best = max(cand, key=_tk); T_avail_ms = _tk(best) * 1000.0
    z = np.load(best, allow_pickle=False)
    spikes = (z["idx"], z["times_ms"]); adj = z["adj_true"].astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    if T_avail_ms > 0 and args.checkpoint_ms > T_avail_ms:
        print(f"clamping checkpoint {args.checkpoint_ms/1000:.0f}k -> available "
              f"{T_avail_ms/1000:.0f}k ms")
        args.checkpoint_ms = T_avail_ms
    print(f"loaded {best.name}  (T={T_avail_ms/1000:.0f}k ms)")

    cp = round(args.checkpoint_ms / dt)
    res, _, rate = stream_moments(
        N=N, sim_time=args.checkpoint_ms, spike_events=spikes, dt=dt,
        lag=lag_est, tau=cfg["tau"], amplitude=cfg["amplitude"],
        sigma_intra=cfg["sigma_intra"], sigma_extra=cfg["sigma_extra"],
        smooth_win=smooth_win, tau_method=cfg["tau_method"],
        checkpoints_samples=[cp], chunk_samples=round(args.chunk_ms / dt),
        seed=1, device=(args.device if args.device == "cuda" else None), lags=lags,
        signal=args.signal)
    snap = res[cp]                                   # {lag: C(lag) numpy}
    print(f"streamed; mean rate {rate:.1f} Hz")

    # per-lag connectivity A_l = C(l) C(0)^-1. GPU (torch) at large N; numpy on CPU
    # (small N=1250 inverses are instant and the CPU env has no torch).
    dev = args.device
    i, j, _ = edge_index(N, args.sample, np.random.default_rng(0))
    g = adj.T[i, j]                                  # true j->i
    P = len(i); nL = len(pos_lags)
    a_ij = np.zeros((P, nL), np.float32); a_ji = np.zeros((P, nL), np.float32)

    if dev == "cuda":
        C0 = torch.tensor(snap[0], device=dev, dtype=torch.float32)
        C0inv = torch.linalg.inv(C0 + 1e-9 * torch.eye(N, device=dev, dtype=torch.float32))
        del C0
        ii = torch.tensor(i, device=dev); jj = torch.tensor(j, device=dev)
        for k, l in enumerate(pos_lags):
            Cl = torch.tensor(snap[l], device=dev, dtype=torch.float32)
            A = Cl @ C0inv; A.fill_diagonal_(0.0)
            a_ij[:, k] = A[ii, jj].cpu().numpy()
            a_ji[:, k] = A[jj, ii].cpu().numpy()
            del Cl, A; torch.cuda.empty_cache()
    else:
        C0inv = np.linalg.inv(snap[0] + 1e-9 * np.eye(N))
        for k, l in enumerate(pos_lags):
            A = snap[l] @ C0inv; np.fill_diagonal(A, 0.0)
            a_ij[:, k] = A[i, j]; a_ji[:, k] = A[j, i]

    est = pos_lags.index(lag_est)                    # column of the estimator lag
    a_est = a_ij[:, est]; arev_est = a_ji[:, est]
    aa = np.abs(a_ij)
    peak_lag_ms = lags_ms[np.argmax(aa, axis=1)]     # timing axis
    asym = np.abs(a_est - arev_est) / (np.abs(a_est) + np.abs(arev_est) + 1e-12)  # symmetry

    # operating point on |A_est|
    tau = float(np.quantile(np.abs(a_est), 1.0 - args.density))
    pred = np.abs(a_est) > tau
    yE, yI, ynone = g > 0, g < 0, g == 0
    pE = pred & (a_est > 0)

    def prof(mask):
        return np.abs(a_ij[mask]).mean(0) if mask.any() else np.full(nL, np.nan)
    profiles = dict(prof_Etrue=prof(pE & yE), prof_Efalse=prof(pE & ynone),
                    prof_allE=prof(yE), prof_allnone=prof(ynone))

    # subsample per-pair rows for the plot (2D + filters)
    rng = np.random.default_rng(1)
    keep = np.arange(P)
    if P > args.keep:
        keep = rng.choice(P, args.keep, replace=False)
    out = dict(net=args.net, N=N, signal=args.signal, lags_ms=lags_ms,
               lag_est_ms=cfg["lag_ms"], delay_ms=cfg["delay"], density=args.density,
               tau=tau, **profiles,
               peak_lag_ms=peak_lag_ms[keep].astype(np.float32),
               asym=asym[keep].astype(np.float32),
               a_est=a_est[keep].astype(np.float32),
               g_sign=np.sign(g[keep]).astype(np.int8),
               pred=pred[keep])

    out_dir = Path(args.out_dir) if args.out_dir else (BASE / "results" / "r8b")
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"r8b_{args.net}_{args.signal}.npz"
    np.savez(fp, **out)
    # quick read-out
    for lab, m in [("true E", pE & yE), ("E false-pos", pE & ynone)]:
        if m.any():
            print(f"  {lab:12s} median peak-lag={np.median(peak_lag_ms[m]):.2f}ms  "
                  f"median asym={np.median(asym[m]):.3f}  n={int(m.sum())}")
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()
