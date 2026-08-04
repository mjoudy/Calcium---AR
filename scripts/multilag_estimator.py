"""
Joint multi-lag estimator — does conditioning on SEVERAL past lags at once
(not one lag at a time) reduce the shared-input false-positive problem?

This is a genuinely different regression from R.8b's multi-lag DIAGNOSTIC.
R.8b fit p separate single-lag regressions and read off the SHAPE of the
resulting profile after the fact. Here, ONE regression predicts x(t) from
[x(t-L1), x(t-L2), ..., x(t-Lp)] stacked together as joint predictors, so the
fit can actually divide a shared driver's influence across lags and cancel it
where the single-lag fit couldn't (this is what Way 2 showed leaks through).

Math (see project discussion): with Gamma(k) := Cov(x(t+k), x(t)) = C(k) for
k>=0 and C(-k)^T for k<0 (stationarity), the block covariance of the stacked
predictors is
    Czz[i,j] = Gamma(Lj - Li)
and the cross-covariance with the target is
    Cyz[i]   = C(Li)
so the joint estimator is  A_joint = Cyz @ inv(Czz), an N x (p*N) matrix. The
block of A_joint belonging to predictor lag L1 is directly comparable to the
existing single-lag estimate at lag L1 (same "predict x(t) from x(t-L1))"
question, just now fit while ALSO controlling for the other lags).

Only needs distinct lags |Li - Lj| for all pairs (i,j) plus the Li themselves
plus 0 -- computed once via stream_moments, same infra as r8b_multilag.py.

Usage (cluster; needs cached ground-truth spikes under --cache-dir):
  python scripts/multilag_estimator.py --net n1250_r4 \
      --lags-ms 1.5 3.0 4.5 6.0 --signal spikes --device cpu \
      --out-dir results/multilag
  python scripts/multilag_estimator.py --net n1250_r4 \
      --lags-ms 1.5 3.0 4.5 6.0 --signal feed --device cpu \
      --out-dir results/multilag
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


def gamma(C, k_ms, lag_to_ms):
    """Gamma(k) = C(k) for k>=0, C(-k)^T for k<0. `C` maps lag-in-samples -> matrix;
    `lag_to_ms` maps lag-in-samples -> ms (both keyed the same as stream_moments' snap)."""
    if k_ms >= 0:
        samp = round(k_ms / lag_to_ms["dt"])
        return C[samp]
    samp = round(-k_ms / lag_to_ms["dt"])
    return C[samp].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, help="preset key, e.g. n1250_r4")
    ap.add_argument("--cache-dir", default=str(BASE / "gt_cache"))
    ap.add_argument("--checkpoint-ms", type=float, default=None,
                     help="default: use the whole cached recording")
    ap.add_argument("--lags-ms", type=float, nargs="+", default=[1.5, 3.0, 4.5, 6.0],
                     help="predictor lags for the JOINT fit (ms)")
    ap.add_argument("--signal", default="feed", choices=["feed", "calcium", "spikes"])
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--sample", type=int, default=20_000_000)
    ap.add_argument("--chunk-ms", type=float, default=5000.0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    torch = None
    if args.device == "cuda":
        import torch
        assert torch.cuda.is_available(), "no CUDA -- use --device cpu"

    cfg = build_cfg(args.net)
    dt = cfg["dt"]; N = cfg["n_excitatory"] + cfg["n_inhibitory"]
    smooth_win = max(5, round(cfg["smooth_window_ms"] / dt))
    L = sorted(args.lags_ms)                                 # predictor lags (ms)
    print(f"net={args.net} N={N} dt={dt} predictor lags(ms)={L}  signal={args.signal}")

    # every distinct |Li-Lj| (i<j), every Li itself, and 0 -- that's all Gamma()
    # ever needs to look up.
    all_ms = sorted({0.0} | set(L) | {abs(a - b) for a in L for b in L})
    lags_samples = sorted({round(v / dt) for v in all_ms})
    print(f"requesting moments at lags(ms)={all_ms} -> samples={lags_samples}")

    # cached ground-truth spikes: longest available recording for this net.
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
    ckpt_ms = args.checkpoint_ms or T_avail_ms
    if ckpt_ms > T_avail_ms:
        ckpt_ms = T_avail_ms
    print(f"loaded {best.name}  using T={ckpt_ms/1000:.0f}k ms of {T_avail_ms/1000:.0f}k available")

    cp = round(ckpt_ms / dt)
    res, _, rate = stream_moments(
        N=N, sim_time=ckpt_ms, spike_events=spikes, dt=dt,
        lag=round(L[0] / dt), tau=cfg["tau"], amplitude=cfg["amplitude"],
        sigma_intra=cfg["sigma_intra"], sigma_extra=cfg["sigma_extra"],
        smooth_win=smooth_win, tau_method=cfg["tau_method"],
        checkpoints_samples=[cp], chunk_samples=round(args.chunk_ms / dt),
        seed=1, device=(args.device if args.device == "cuda" else None),
        lags=lags_samples, signal=args.signal)
    snap = res[cp]                                             # {lag_samples: C}
    print(f"streamed; mean rate {rate:.1f} Hz")

    ctx = {"dt": dt}
    def C(k_ms):
        return gamma(snap, k_ms, ctx)

    p = len(L)
    xp = np if args.device == "cpu" else torch
    def to_dev(M):
        return M if args.device == "cpu" else torch.tensor(M, device="cuda", dtype=torch.float32)
    inv = np.linalg.inv if args.device == "cpu" else torch.linalg.inv
    eye = np.eye if args.device == "cpu" else (lambda n: torch.eye(n, device="cuda"))

    # assemble the (p*N, p*N) block covariance of the stacked predictors, and
    # the (N, p*N) cross-covariance with the current-time target.
    Czz = np.zeros((p * N, p * N), dtype=np.float64)
    for i in range(p):
        for j in range(p):
            Czz[i*N:(i+1)*N, j*N:(j+1)*N] = C(L[j] - L[i])
    Cyz = np.concatenate([C(L[i]) for i in range(p)], axis=1)   # (N, p*N)

    reg = 1e-9
    if args.device == "cpu":
        A_joint = Cyz @ np.linalg.inv(Czz + reg * np.eye(p * N))       # (N, p*N)
    else:
        Czz_t = torch.tensor(Czz, device="cuda", dtype=torch.float32)
        Cyz_t = torch.tensor(Cyz, device="cuda", dtype=torch.float32)
        A_joint_t = Cyz_t @ torch.linalg.inv(Czz_t + reg * torch.eye(p*N, device="cuda"))
        A_joint = A_joint_t.cpu().numpy()

    # baseline: plain single-lag OLS at the FIRST predictor lag (same question,
    # not conditioned on the others) -- the thing we're testing against.
    A_single = C(L[0]) @ np.linalg.inv(C(0.0) + reg * np.eye(N))
    A_joint_L1 = A_joint[:, 0:N]                                 # L[0]-block of the joint fit
    for A in (A_single, A_joint_L1):
        np.fill_diagonal(A, 0.0)

    # -------- false-positive comparison, same procedure as fig_way2.py -------- #
    i, j, _ = edge_index(N, args.sample, np.random.default_rng(0))
    g = adj.T[i, j]                                              # true j->i
    yE, ynone = g > 0, g == 0

    def report(name, A):
        a = A[i, j]; aa = np.abs(a)
        tau = np.quantile(aa, 1.0 - args.density)
        pred = aa > tau
        pE = pred & (a > 0)
        tp, fp = pE & yE, pE & ynone
        prec = tp.sum() / max(pred.sum(), 1)
        fp_mean = aa[fp].mean() if fp.any() else float("nan")
        fp_med = np.median(aa[fp]) if fp.any() else float("nan")
        tp_mean = aa[tp].mean() if tp.any() else float("nan")
        print(f"  {name:14s} n_tp={int(tp.sum()):7d} n_fp={int(fp.sum()):7d} "
              f"E-precision={prec:.3f}  FP mean|A|={fp_mean:.4g} med={fp_med:.4g}  "
              f"TP mean|A|={tp_mean:.4g}")
        stats = dict(n_tp=int(tp.sum()), n_fp=int(fp.sum()), precision=float(prec),
                     fp_mean=float(fp_mean), fp_median=float(fp_med), tp_mean=float(tp_mean))
        return stats, tp, fp

    print(f"\n--- false-positive comparison (signal={args.signal}, "
          f"L[0]={L[0]}ms, joint predictors={L}) ---")
    r_single, tp_single, fp_single = report("single-lag", A_single)
    r_joint, tp_joint, fp_joint = report("joint multi-lag", A_joint_L1)

    # -------- is the improvement specifically about shared input? -------- #
    # 1-hop common-presynaptic-driver count for every sampled pair (i,j), same
    # construction as fig_way2.py / fig_way2_motifs.py, computed once from the
    # already-loaded ground-truth adjacency (cheap: boolean matmul at N=1250).
    Bfull = (adj != 0)
    driver_count = (Bfull[:, i] & Bfull[:, j]).sum(0)            # shared over full N

    fixed = fp_single & ~fp_joint      # was a false positive, joint no longer flags it
    persisted = fp_single & fp_joint   # false positive under both
    new_tp = tp_joint & ~tp_single     # newly gained true positives (sanity check)

    def dc_stats(mask, label):
        n = int(mask.sum())
        if n == 0:
            print(f"  {label:34s} n=0"); return dict(n=0, mean=float("nan"), median=float("nan"))
        m, med = float(driver_count[mask].mean()), float(np.median(driver_count[mask]))
        print(f"  {label:34s} n={n:7d}  mean driver_count={m:.2f}  median={med:.1f}")
        return dict(n=n, mean=m, median=med)

    print(f"\n--- shared-driver exposure: is the fix specifically about shared input? ---")
    dc_all_fp = dc_stats(fp_single, "ALL single-lag false positives")
    dc_fixed = dc_stats(fixed, "  -> FIXED by joint estimator")
    dc_persisted = dc_stats(persisted, "  -> PERSISTED under joint estimator")
    dc_new_tp = dc_stats(new_tp, "newly-gained true positives (sanity check)")
    if dc_fixed["n"] and dc_persisted["n"]:
        print(f"\n  fixed/persisted mean driver_count ratio: "
              f"{dc_fixed['mean']/max(dc_persisted['mean'],1e-9):.2f}x "
              f"(>1 means the fix IS concentrated on high-shared-input false positives)")

    out = dict(net=args.net, N=N, signal=args.signal, lags_ms=np.array(L),
               density=args.density, checkpoint_ms=ckpt_ms,
               single_precision=r_single["precision"], joint_precision=r_joint["precision"],
               single_fp_mean=r_single["fp_mean"], joint_fp_mean=r_joint["fp_mean"],
               single_fp_median=r_single["fp_median"], joint_fp_median=r_joint["fp_median"],
               dc_fixed_n=dc_fixed["n"], dc_fixed_mean=dc_fixed["mean"], dc_fixed_median=dc_fixed["median"],
               dc_persisted_n=dc_persisted["n"], dc_persisted_mean=dc_persisted["mean"],
               dc_persisted_median=dc_persisted["median"],
               dc_new_tp_n=dc_new_tp["n"], dc_new_tp_mean=dc_new_tp["mean"],
               dc_all_fp_mean=dc_all_fp["mean"])
    out_dir = Path(args.out_dir) if args.out_dir else (BASE / "results" / "multilag")
    out_dir.mkdir(parents=True, exist_ok=True)
    fp_path = out_dir / f"multilag_{args.net}_{args.signal}.npz"
    np.savez(fp_path, **out)
    print(f"\nwrote {fp_path}")


if __name__ == "__main__":
    main()
