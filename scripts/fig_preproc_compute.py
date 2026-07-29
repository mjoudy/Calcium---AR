"""
Effect of preprocessing — COMPUTE (refresh at scale, consistent metrics).

Compares connectivity recovery from:
  RAW calcium  : OLS on the fluorescence directly (no smoothing / deconvolution).
  FEED         : OLS on the deconvolved feed (SG smooth -> tau -> deriv + C/tau) —
                 the normal pipeline.

Both use the SAME cached spikes and the SAME calcium (seeded noise), so the only
difference is the deconvolution step. Scored with the R.1-R.8 scorer
(analyze_run.score) so the numbers sit on the same axis as the rest.

Writes <out-dir>/preproc_<net>.npz  (metrics + class-conditional weight CDFs).
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
from calcium_ar.solvers.from_moments import ols_from_moments
from analyze_run import edge_index, score
from wrapup_run import build_cfg
BASE = Path(os.environ.get("CALCIUM_AR_WORKDIR", ROOT))


def solve(net_key, cfg, spikes, N, T_ms, dt, smooth_win, lag, device, raw):
    cp = round(T_ms / dt)
    res, _, _ = stream_moments(
        N=N, sim_time=T_ms, spike_events=spikes, dt=dt, lag=lag,
        tau=cfg["tau"], amplitude=cfg["amplitude"], sigma_intra=cfg["sigma_intra"],
        sigma_extra=cfg["sigma_extra"], smooth_win=smooth_win,
        tau_method=cfg["tau_method"], checkpoints_samples=[cp],
        chunk_samples=round(5000 / dt), seed=1,
        device=(device if device == "cuda" else None), raw_calcium=raw)
    Cxx, Cyx = res[cp]
    return ols_from_moments(Cxx, Cyx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--cache-dir", default=str(BASE / "gt_cache"))
    ap.add_argument("--checkpoint-ms", type=float, default=500_000.0)
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--sample", type=int, default=20_000_000)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = build_cfg(args.net)
    dt = cfg["dt"]; N = cfg["n_excitatory"] + cfg["n_inhibitory"]
    smooth_win = max(5, round(cfg["smooth_window_ms"] / dt))
    lag = max(1, round(cfg["lag_ms"] / dt))

    cache = Path(args.cache_dir)
    cand = sorted(cache.glob(f"{args.net}_seed1_*.npz"))
    if not cand:
        raise SystemExit(f"no cached spikes {args.net}_seed1_*.npz under {cache}")
    z = np.load(cand[-1], allow_pickle=False)
    spikes = (z["idx"], z["times_ms"]); adj = z["adj_true"].astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    print(f"{cand[-1].name}  N={N}  T={args.checkpoint_ms/1000:.0f}k ms")

    i, j, _ = edge_index(N, args.sample, np.random.default_rng(0))
    g = adj.T[i, j]
    yE, yI, ynone = g > 0, g < 0, g == 0

    results = {}
    cdfs = {}
    for raw, name in [(True, "raw"), (False, "feed")]:
        print(f"=== {name} ===", flush=True)
        A = solve(args.net, cfg, spikes, N, args.checkpoint_ms, dt, smooth_win,
                  lag, args.device, raw)
        a = A[i, j]
        m = score(a, g, args.density)
        results[name] = m
        print(f"  corr={m['corr']:.3f}  AUC_E={m['auc_E']:.3f}  "
              f"E_rec={m['E_rec']:.3f}  I_rec={m['I_rec']:.3f}", flush=True)
        # class-conditional |weight| CDFs (subsample), normalised per estimate
        an = np.abs(a) / (np.abs(a).std() + 1e-12)
        rng = np.random.default_rng(1)
        for cls, mask in [("E", yE), ("I", yI), ("none", ynone)]:
            v = an[mask]
            if len(v) > 100_000:
                v = v[rng.choice(len(v), 100_000, replace=False)]
            cdfs[f"{name}_{cls}"] = np.sort(v).astype(np.float32)

    metrics = ["corr", "spearman_na", "auc_roc", "auc_E", "auc_I", "E_rec", "I_rec"]
    out = dict(net=args.net, N=N, T_ms=args.checkpoint_ms, **cdfs)
    for name in ("raw", "feed"):
        for k in ("corr", "roc_auc", "auc_E", "auc_I", "E_rec", "E_prec", "I_rec"):
            out[f"{name}_{k}"] = float(results[name][k])

    out_dir = Path(args.out_dir) if args.out_dir else (BASE / "results" / "preproc")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"preproc_{args.net}.npz", **out)
    print(f"\nwrote {out_dir}/preproc_{args.net}.npz")


if __name__ == "__main__":
    main()
