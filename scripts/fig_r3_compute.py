"""
R.3 (lag & delay) — COMPUTE stage.

Causal-validity test: does the regression lag that maximises accuracy track the
TRUE synaptic delay of the network? For each true delay D we simulate, build the
deconvolved feed once, then sweep the regression lag and score OLS.

Writes r3_data.npz: per delay, (lag_ms, AUC, corr); plus the argmax lag per delay.

Usage:
  python scripts/fig_r3_compute.py --net n1250ai --T-ms 60000 \
      --out $(ws_find calcium_ar)/results/fig_data/r3_data.npz
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
from calcium_ar.solvers.from_moments import ols_from_moments
from wrapup_run import build_cfg

SMOOTH_MS, TAU, AMP, SIG_IN, SIG_EX = 3.1, 100.0, 1.0, 0.01, 0.05


def deconv_feed(idx, tms, N, dt, T_ms, chunk, seed):
    """Build the full deconvolved feed (N, T) float32, chunk-by-chunk so only one
    full-size array is ever held."""
    rng = np.random.default_rng(seed)
    T = int(T_ms / dt)
    w = max(5, round(SMOOTH_MS / dt)); w = w if w % 2 else w + 1
    alpha = float(np.exp(-dt / TAU))
    b, a = np.array([1.0], np.float32), np.array([1.0, -alpha], np.float32)
    zi = np.zeros((N, 1), np.float32)
    feed = np.empty((N, T), np.float32)
    tau_est = None
    t0 = 0
    while t0 < T:
        t1 = min(t0 + chunk, T); L = t1 - t0
        spk = np.zeros((N, L), np.float32)
        sel = (tms >= t0 * dt) & (tms < t1 * dt)
        s = np.clip(np.round(tms[sel] / dt).astype(np.int64) - t0, 0, L - 1)
        np.add.at(spk, (idx[sel], s), np.float32(1.0))
        inp = AMP * spk + rng.standard_normal((N, L), dtype=np.float32) * np.float32(SIG_IN)
        C, zi = lfilter(b, a, inp, axis=1, zi=zi)
        F = C + rng.standard_normal((N, L), dtype=np.float32) * np.float32(SIG_EX)
        if tau_est is None:
            tau_est = np.atleast_1d(estimate_tau_robust(F, window_length=w,
                                                        method="ransac", dt=dt))
        sm = smooth_signal(F, window_length=w, deriv=0, delta=dt)
        dv = smooth_signal(F, window_length=w, deriv=1, delta=dt)
        feed[:, t0:t1] = dv + sm / (tau_est[:, None] if tau_est.ndim else tau_est)
        t0 = t1
    return feed


def score_at_lag(feed, adj, lag):
    xp = feed[:, :-lag]; xn = feed[:, lag:]
    xp = xp - xp.mean(1, keepdims=True); xn = xn - xn.mean(1, keepdims=True)
    n = xp.shape[1]
    Cxx = (xp @ xp.T) / n; Cyx = (xn @ xp.T) / n
    A = ols_from_moments(Cxx.astype(np.float64), Cyx.astype(np.float64))
    N = A.shape[0]; m = ~np.eye(N, dtype=bool)
    g = adj.T[m].ravel(); v = A[m].ravel()
    return (float(roc_auc_score((g != 0).astype(int), np.abs(v))),
            float(np.corrcoef(v, g)[0, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="n1250ai")
    ap.add_argument("--T-ms", type=float, default=60_000.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--delays", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0, 5.0])
    ap.add_argument("--lags-ms", type=float, nargs="+",
                    default=[0.5, 1, 1.5, 2, 3, 5, 8, 12, 20])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = build_cfg(args.net); dt = cfg["dt"]
    save = dict(net=args.net, delays=np.array(args.delays), lags_ms=np.array(args.lags_ms))
    opt_lag = []
    for D in args.delays:
        print(f"delay={D} ms: simulating ...", flush=True)
        net = BrunelNetwork(n_excitatory=cfg["n_excitatory"], n_inhibitory=cfg["n_inhibitory"],
                            epsilon=cfg["epsilon"], g=cfg["g"], eta=cfg["eta"], J_ex=cfg["J_ex"],
                            delay=D, V_reset=cfg["V_reset"], sim_time=args.T_ms, dt=dt,
                            n_threads=cfg["n_threads"], seed=args.seed)
        net.build(); net.run(densify=False)
        idx = net.get_spike_events()[0].astype(np.int64); tms = net.get_spike_events()[1]
        adj = net.get_adjacency(); np.fill_diagonal(adj, 0.0); N = adj.shape[0]
        feed = deconv_feed(idx, tms, N, dt, args.T_ms, args.chunk, args.seed)
        aucs, corrs = [], []
        for lm in args.lags_ms:
            lag = max(1, round(lm / dt))
            auc, corr = score_at_lag(feed, adj, lag)
            aucs.append(auc); corrs.append(corr)
            print(f"    lag={lm:<5g} ms  AUC={auc:.3f}  corr={corr:.3f}", flush=True)
        del feed
        aucs = np.array(aucs)
        save[f"auc_D{D}"] = aucs
        save[f"corr_D{D}"] = np.array(corrs)
        opt_lag.append(args.lags_ms[int(np.argmax(aucs))])
        print(f"  -> optimal lag {opt_lag[-1]} ms (true delay {D} ms)", flush=True)
    save["opt_lag"] = np.array(opt_lag)
    np.savez(args.out, **save)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
