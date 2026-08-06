"""
R.2 (calcium-observation section) — COMPUTE stage.

From ONE simulation (sparse spikes), build five signal chains and score OLS on
each with two measures (ROC-AUC + correlation), at a FIXED recording length so
differences reflect the OBSERVATION, not the amount of data:

  spikes        : regress on binned spike counts (no calcium) — the ceiling
  deconv_tau    : dye tau swept, camera FIXED at FIXED_CAM_MS -> deconvolve -> feed
  deconv_rate   : camera dt swept, dye tau FIXED at FIXED_TAU_MS -> deconvolve -> feed
  raw_tau       : dye tau swept, camera FIXED,  NO deconvolution
  raw_rate      : camera dt swept, dye tau FIXED, NO deconvolution

Two physically distinct knobs, swept ONE AT A TIME (tau = dye/indicator decay
kinetics; camera dt = recording sampling interval) rather than collapsed onto a
shared dt/tau ratio axis — a combined ratio axis makes it impossible to tell
which physical change (dye vs. camera) is driving a given point. Each sweep
also still records eff_dt/tau so the "does only the ratio matter?" collapse
can be plotted separately (fig_r2_plot.py's secondary panel) for those who
want it, without it being the primary read.

Ranges are pushed past the previous endpoints on both sweeps so the curves
visibly reach their floor/plateau instead of stopping mid-slope.

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

# Fixed value held constant on the OTHER sweep, so each sweep changes exactly
# one physical knob at a time:
FIXED_TAU_MS = 100.0   # dye decay used while sweeping camera rate
FIXED_CAM_MS = 33.0    # ~30 Hz camera used while sweeping dye tau (typical
                        # calcium-imaging frame rate; not an infinitely-fast camera)


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
    """Returns (x_native, x_ratio, auc, corr).

    x_native is the ONE physical quantity this sweep actually varies (camera
    frame interval in ms for a rate sweep, dye tau in ms for a tau sweep) —
    the primary, single-variable x-axis. x_ratio = eff_dt/tau is kept
    alongside it only for the secondary "does the ratio alone explain it?"
    collapse view.
    """
    rng = np.random.default_rng(rng_seed)
    if kind == "spikes":
        eff_dt = param; lag = max(1, round(LAG_MS / eff_dt))
        Cxx, Cyx = accumulate(iter_spike_bin(idx, tms, N, param, T_ms, chunk), N, lag)
        x_native = param; x_ratio = param
    else:
        deconv = kind.startswith("deconv")
        if kind.endswith("tau"):
            # camera held FIXED at a realistic (not infinitely-fast) rate,
            # so we can sweep tau both above and below the camera's own dt.
            tau = param; frame_ms = FIXED_CAM_MS; eff_dt = FIXED_CAM_MS
            x_native = tau
        else:                                          # rate sweep, tau FIXED
            tau = FIXED_TAU_MS; frame_ms = param; eff_dt = param
            x_native = frame_ms
        lag = max(1, round(LAG_MS / eff_dt))
        Cxx, Cyx = accumulate(
            iter_calcium(idx, tms, N, dt, tau, T_ms, deconv, frame_ms, chunk, rng), N, lag)
        x_ratio = eff_dt / tau
    auc, corr = score(Cxx, Cyx, adj)
    return x_native, x_ratio, auc, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="n1250ai")
    ap.add_argument("--T-ms", type=float, default=100_000.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=20000, help="frames per chunk")
    ap.add_argument("--taus", type=float, nargs="+",
                    # camera fixed at FIXED_CAM_MS=33: values below 33 cross into
                    # "dye decays faster than the camera can see" territory;
                    # values up to 1600 extend into "very slow, over-smoothed
                    # dye" territory, so both ends of the curve reach a plateau.
                    default=[0.5, 1, 2, 5, 10, 20, 50, 100, 200, 400, 800, 1600])
    ap.add_argument("--frames", type=float, nargs="+",
                    # tau fixed at FIXED_TAU_MS=100: extended past 33 out to
                    # 1000 ms (~1 fps) so the curve visibly bottoms out instead
                    # of stopping mid-slope.
                    default=[0.1, 0.5, 1, 2, 5, 10, 20, 33, 50, 100, 200, 500, 1000])
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
        x_native, x_ratio, auc, corr = run(
            kind, p, idx, tms, N, dt, adj, args.T_ms, args.chunk, args.seed)
        out[kind].append((x_native, x_ratio, auc, corr))
        print(f"  {kind:12s} param={p:<6g} x={x_native:.4g}  "
              f"(dt/tau={x_ratio:.4g})  AUC={auc:.3f}  corr={corr:.3f}", flush=True)

    save = dict(net=args.net, N=N, T_ms=args.T_ms,
                fixed_tau_ms=FIXED_TAU_MS, fixed_cam_ms=FIXED_CAM_MS)
    for k, v in out.items():
        arr = np.array(sorted(v))
        save[f"{k}_x"], save[f"{k}_ratio"] = arr[:, 0], arr[:, 1]
        save[f"{k}_auc"], save[f"{k}_corr"] = arr[:, 2], arr[:, 3]
    np.savez(args.out, **save)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
