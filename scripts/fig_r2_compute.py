"""
R.2 (calcium-observation section) — COMPUTE stage.

From ONE simulation (sparse spikes), build five signal chains and score OLS on
each with two measures (ROC-AUC + correlation), at a FIXED recording length so
differences reflect the OBSERVATION, not the amount of data:

  spikes        : regress on binned spike counts (no calcium) — the ceiling
  deconv_tau    : calcium (sweep dye tau) -> deconvolve      -> feed
  deconv_rate   : calcium -> downsample (sweep camera dt)    -> deconvolve -> feed
  raw_tau       : calcium (sweep tau), NO deconvolution
  raw_rate      : calcium -> downsample (sweep dt), NO deconvolution

The tau- and rate-swept curves share the x-axis dt/tau (the calcium "blur"): if
only the ratio matters they overlap; the camera (rate) curve may sit lower
because a slower camera also loses frames and merges spikes.

Writes a small r2_data.npz (KB). Reuses cached spikes via --cache-dir if present.

Usage (cluster or local, N=1250 is cheap):
  python scripts/fig_r2_compute.py --net n1250ai --T-ms 100000 \
      --out $(ws_find calcium_ar)/results/fig_data/r2_data.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from calcium_ar.simulation.brunel_network import BrunelNetwork
from calcium_ar.preprocessing.signal_utils import smooth_signal
from calcium_ar.preprocessing.tau_estimation import estimate_tau_robust
from calcium_ar.experiments.streaming import MomentAccumulator
from calcium_ar.solvers.from_moments import ols_from_moments
from wrapup_run import build_cfg

LAG_MS = 2.0
SMOOTH_MS = 3.1
AMP, SIG_IN, SIG_EX = 1.0, 0.01, 0.05


def smooth_win(eff_dt):
    w = max(5, round(SMOOTH_MS / eff_dt))
    return w if w % 2 == 1 else w + 1


def score(Cxx, Cyx, adj):
    A = ols_from_moments(Cxx, Cyx)
    N = A.shape[0]; m = ~np.eye(N, dtype=bool)
    g = adj.T[m].ravel(); a = A[m].ravel()
    return (float(roc_auc_score((g != 0).astype(int), np.abs(a))),
            float(np.corrcoef(a, g)[0, 1]))


# --------------------------------------------------------------------------- #
# feed-chunk generators (each yields feed columns at its effective dt)         #
# --------------------------------------------------------------------------- #

def iter_spike_bin(idx, tms, N, bin_ms, T_ms, chunk):
    nb = int(T_ms // bin_ms)
    t0 = 0
    while t0 < nb:
        t1 = min(t0 + chunk, nb); L = t1 - t0
        spk = np.zeros((N, L), dtype=np.float32)
        sel = (tms >= t0 * bin_ms) & (tms < t1 * bin_ms)
        b = (np.floor(tms[sel] / bin_ms).astype(np.int64) - t0)
        np.add.at(spk, (idx[sel].astype(np.int64), b), np.float32(1.0))
        yield spk
        t0 = t1


def iter_calcium(idx, tms, N, dt, tau, T_ms, deconv, frame_ms, chunk, rng):
    """Fine calcium via AR(1); optional downsample to frame_ms; optional deconv.
    Yields feed columns at eff_dt = frame_ms (or dt)."""
    eff_dt = frame_ms if frame_ms else dt
    r = max(1, int(round(eff_dt / dt)))               # downsample factor
    win = smooth_win(eff_dt)
    alpha = float(np.exp(-dt / tau))
    b, a = np.array([1.0], np.float32), np.array([1.0, -alpha], np.float32)
    zi = np.zeros((N, 1), np.float32)
    tau_est = [None]
    T_fine = int(T_ms / dt)
    fine_chunk = chunk * r
    t0 = 0
    while t0 < T_fine:
        t1 = min(t0 + fine_chunk, T_fine); L = t1 - t0
        spk = np.zeros((N, L), np.float32)
        sel = (tms >= t0 * dt) & (tms < t1 * dt)
        s = (np.round(tms[sel] / dt).astype(np.int64) - t0)
        s = np.clip(s, 0, L - 1)
        np.add.at(spk, (idx[sel].astype(np.int64), s), np.float32(1.0))
        inp = AMP * spk + rng.standard_normal((N, L), dtype=np.float32) * np.float32(SIG_IN)
        C, zi = lfilter(b, a, inp, axis=1, zi=zi)
        F = C + rng.standard_normal((N, L), dtype=np.float32) * np.float32(SIG_EX)
        if r > 1:                                     # downsample (camera)
            nkeep = (F.shape[1] // r) * r
            F = F[:, :nkeep:r]
        if F.shape[1] == 0:
            t0 = t1; continue
        if not deconv:
            yield F - F.mean(1, keepdims=True)
        else:
            if tau_est[0] is None:
                tau_est[0] = estimate_tau_robust(F, window_length=win,
                                                 method="ransac", dt=eff_dt)
            sm = smooth_signal(F, window_length=win, deriv=0, delta=eff_dt)
            dv = smooth_signal(F, window_length=win, deriv=1, delta=eff_dt)
            te = np.atleast_1d(tau_est[0])
            yield dv + sm / (te[:, None] if te.ndim else te)
        t0 = t1


def accumulate(feed_iter, N, lag):
    acc = MomentAccumulator(N, lag)
    for fc in feed_iter:
        acc.add(np.asarray(fc, dtype=np.float64))
    return acc.snapshot()


def run(kind, param, idx, tms, N, dt, adj, T_ms, chunk, rng_seed):
    rng = np.random.default_rng(rng_seed)
    if kind == "spikes":
        eff_dt = param; lag = max(1, round(LAG_MS / eff_dt))
        Cxx, Cyx = accumulate(iter_spike_bin(idx, tms, N, param, T_ms, chunk), N, lag)
        x = param
    else:
        deconv = kind.startswith("deconv")
        if kind.endswith("tau"):
            tau = param; frame_ms = None; eff_dt = dt
        else:                                          # rate sweep, tau fixed 100
            tau = 100.0; frame_ms = param; eff_dt = param
        lag = max(1, round(LAG_MS / eff_dt))
        Cxx, Cyx = accumulate(
            iter_calcium(idx, tms, N, dt, tau, T_ms, deconv, frame_ms, chunk, rng), N, lag)
        x = eff_dt / tau
    auc, corr = score(Cxx, Cyx, adj)
    return x, auc, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="n1250ai")
    ap.add_argument("--T-ms", type=float, default=100_000.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=20000, help="frames per chunk")
    ap.add_argument("--taus", type=float, nargs="+",
                    # spans dt/tau = 0.1 .. 0.000125 (dt=0.1), so the dye sweep
                    # overlaps the camera sweep across the whole x-axis
                    default=[1, 2, 5, 10, 25, 50, 100, 200, 400, 800])
    ap.add_argument("--frames", type=float, nargs="+",
                    default=[0.1, 0.5, 1, 2, 5, 10, 20, 33])
    ap.add_argument("--spike-bins", type=float, nargs="+",
                    default=[0.5, 1, 2, 5, 10, 20, 50])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = build_cfg(args.net); dt = cfg["dt"]
    print(f"simulating {args.net} for {args.T_ms:.0f} ms ...", flush=True)
    net = BrunelNetwork(n_excitatory=cfg["n_excitatory"], n_inhibitory=cfg["n_inhibitory"],
                        epsilon=cfg["epsilon"], g=cfg["g"], eta=cfg["eta"], J_ex=cfg["J_ex"],
                        delay=cfg["delay"], V_reset=cfg["V_reset"], sim_time=args.T_ms,
                        dt=dt, n_threads=cfg["n_threads"], seed=args.seed)
    net.build(); net.run(densify=False)
    idx, tms = net.get_spike_events()
    idx = idx.astype(np.int64)
    adj = net.get_adjacency(); np.fill_diagonal(adj, 0.0)
    N = adj.shape[0]

    out = {k: [] for k in ["spikes", "deconv_tau", "deconv_rate", "raw_tau", "raw_rate"]}
    plan = ([("spikes", b) for b in args.spike_bins]
            + [("deconv_tau", t) for t in args.taus]
            + [("raw_tau", t) for t in args.taus]
            + [("deconv_rate", f) for f in args.frames]
            + [("raw_rate", f) for f in args.frames])
    for kind, p in plan:
        x, auc, corr = run(kind, p, idx, tms, N, dt, adj, args.T_ms, args.chunk, args.seed)
        out[kind].append((x, auc, corr))
        print(f"  {kind:12s} param={p:<6g} x={x:.4g}  AUC={auc:.3f}  corr={corr:.3f}",
              flush=True)

    save = dict(net=args.net, N=N, T_ms=args.T_ms)
    for k, v in out.items():
        arr = np.array(sorted(v))
        save[f"{k}_x"], save[f"{k}_auc"], save[f"{k}_corr"] = arr[:, 0], arr[:, 1], arr[:, 2]
    np.savez(args.out, **save)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
