"""
R.2 (calcium-observation section) — COMPUTE stage.

From ONE simulation (sparse spikes), build five signal chains and score OLS on
each with the full connectivity metric set (ROC-AUC + correlation, both
threshold-free, plus density-quantile precision / recall / F1 and the
excitatory- vs inhibitory-source recall split -- see scripts/r2_metrics.py), at
a FIXED recording length so differences reflect the OBSERVATION, not the amount
of data:

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

Also dumps the per-point OLS estimate A (float32 .npy) plus a _meta.npz
(adj, n_exc) into <out-without-.npz>_estimates/ (disable with
--no-save-estimates). scripts/fig_r2_rescore.py re-derives the metric set from
those without re-running this ~1h sweep -- so adding / retuning a metric later
is a seconds-long rescore, not a cluster job.

Usage (cluster or local, N=1250 is cheap):
  python scripts/fig_r2_compute.py --net n1250ai --T-ms 100000 \
      --out $(ws_find calcium_ar)/results/fig_data/r2_data.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from calcium_ar.simulation.brunel_network import BrunelNetwork
from calcium_ar.preprocessing.signal_utils import smooth_signal
from calcium_ar.preprocessing.tau_estimation import estimate_tau_robust
from calcium_ar.experiments.streaming import MomentAccumulator
from calcium_ar.solvers.from_moments import ols_from_moments
from r2_metrics import METRICS, metrics_from_A
from wrapup_run import build_cfg

DENSITY = 0.10   # operating-point density for precision/recall/F1 (~Brunel epsilon)

LAG_MS = 2.0
SMOOTH_MS = 3.1
AMP, SIG_IN, SIG_EX = 1.0, 0.01, 0.05
CHUNK_MEM_BUDGET_BYTES = 400_000_000   # cap per-chunk (N, L) float32 array size

# Fixed value held constant on the OTHER sweep, so each sweep changes exactly
# one physical knob at a time:
FIXED_TAU_MS = 100.0   # dye decay used while sweeping camera rate
FIXED_CAM_MS = 33.0    # ~30 Hz camera used while sweeping dye tau (typical
                        # calcium-imaging frame rate; not an infinitely-fast camera)


def smooth_win(eff_dt):
    w = max(5, round(SMOOTH_MS / eff_dt))
    return w if w % 2 == 1 else w + 1


def score(Cxx, Cyx, adj, n_exc, density=DENSITY):
    """Solve OLS from the moments, then score the estimate. Returns (A, mets)
    where mets is a dict keyed by scripts/r2_metrics.METRICS. A is handed back
    so the caller can dump it for later rescoring."""
    A = ols_from_moments(Cxx, Cyx)
    return A, metrics_from_A(A, adj, n_exc, density)


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


def iter_calcium(idx, tms, N, dt, tau, T_ms, deconv, frame_ms, chunk, rng, phase=0):
    """Fine calcium via AR(1); optional downsample to frame_ms; optional deconv.
    Yields feed columns at eff_dt = frame_ms (or dt).

    phase (in FINE samples, 0 <= phase < r): which fine sample the camera's
    first kept frame lands on, e.g. phase=0 is "starts at t=0" (the original,
    only behaviour before this was added), phase=1 is "starts 1 fine-dt
    later", etc. Used to check whether the dropped in-between samples carry
    extra information when reused (see docs/experiments/notebook.md)."""
    eff_dt = frame_ms if frame_ms else dt
    r = max(1, int(round(eff_dt / dt)))               # downsample factor
    win = smooth_win(eff_dt)
    alpha = float(np.exp(-dt / tau))
    b, a = np.array([1.0], np.float32), np.array([1.0, -alpha], np.float32)
    zi = np.zeros((N, 1), np.float32)
    tau_est = [None]
    T_fine = int(T_ms / dt)
    # `chunk` is sized in OUTPUT (post-downsample) samples, so the fine-
    # resolution array for one iteration is chunk*r samples wide -- fine for
    # small r, but at a heavily downsampled camera (e.g. 33 ms -> r=330) that
    # is chunk*r = 6.6M samples, i.e. a (N, 6.6M) float32 array (~33 GB at
    # N=1250). Cap the FINE chunk directly at a fixed memory budget instead,
    # independent of r: more (smaller) iterations at high r, same result,
    # since the moment accumulation is exact regardless of chunking.
    # Also make sure each chunk yields enough POST-downsample samples for the
    # smoothing window (win), not just >0 -- at a heavily downsampled camera
    # (large r) a trailing/remainder chunk can otherwise end up with fewer
    # output samples than win, which savgol_filter rejects outright.
    max_fine_chunk = max(r * win, CHUNK_MEM_BUDGET_BYTES // (max(N, 1) * 4))
    fine_chunk = min(chunk * r, max_fine_chunk)
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
            # phase-correct across chunk boundaries even when a chunk length
            # isn't a multiple of r: convert the GLOBAL phase into this
            # chunk's LOCAL starting offset.
            start_local = (phase - t0) % r
            F = F[:, start_local::r]
        if F.shape[1] < win:
            # too few post-downsample samples for this chunk to smooth at all
            # (only possible on a short trailing remainder) -- dropping it
            # loses a negligible fraction of one (N, T) moment accumulation,
            # not worth complicating the chunker to avoid.
            t0 = t1; continue
        if not deconv:
            # NOT F - F.mean(1, keepdims=True): per-chunk local centering here
            # silently detrends slow shared drift before MomentAccumulator does
            # its own correct GLOBAL centering at snapshot time -- double
            # centering that only ever touched this (raw) branch, not the
            # deconvolved one, and not what stream_moments() (the actual
            # production pipeline fig_preproc/fig_best use) does either. It was
            # inflating the raw arm's correlation relative to the real method.
            yield F
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


def run(kind, param, idx, tms, N, dt, adj, T_ms, chunk, rng_seed, n_exc,
        density=DENSITY, fixed_cam_ms=FIXED_CAM_MS, fixed_tau_ms=FIXED_TAU_MS):
    """Returns (x_native, x_ratio, mets, A) — mets keyed by r2_metrics.METRICS.

    x_native is the ONE physical quantity this sweep actually varies (camera
    frame interval in ms for a rate sweep, dye tau in ms for a tau sweep) —
    the primary, single-variable x-axis. x_ratio = eff_dt/tau is kept
    alongside it only for the secondary "does the ratio alone explain it?"
    collapse view.

    fixed_cam_ms / fixed_tau_ms override the module-level defaults (33 ms /
    100 ms) — e.g. to hold the camera at the idealised 0.1 ms used everywhere
    else in this thesis instead of a "realistic" 33 ms, for a tau sweep that's
    apples-to-apples with every other figure rather than a separate regime.
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
            tau = param; frame_ms = fixed_cam_ms; eff_dt = fixed_cam_ms
            x_native = tau
        else:                                          # rate sweep, tau FIXED
            tau = fixed_tau_ms; frame_ms = param; eff_dt = param
            x_native = frame_ms
        lag = max(1, round(LAG_MS / eff_dt))
        Cxx, Cyx = accumulate(
            iter_calcium(idx, tms, N, dt, tau, T_ms, deconv, frame_ms, chunk, rng), N, lag)
        x_ratio = eff_dt / tau
    A, mets = score(Cxx, Cyx, adj, n_exc, density)
    return x_native, x_ratio, mets, A


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
    ap.add_argument("--kinds", nargs="+",
                    default=["spikes", "deconv_tau", "raw_tau", "deconv_rate", "raw_rate"],
                    choices=["spikes", "deconv_tau", "raw_tau", "deconv_rate", "raw_rate"],
                    help="restrict which of the 5 arms to run (default: all, "
                         "matching every past use of this script) -- e.g. "
                         "'--kinds spikes deconv_rate raw_rate' skips the slower "
                         "tau sweep when only the camera-rate panel is wanted")
    ap.add_argument("--fixed-cam-ms", type=float, default=FIXED_CAM_MS,
                    help="camera dt held fixed while tau is swept (default: "
                         "33 ms, a realistic frame rate; pass 0.1 for the "
                         "idealised camera used everywhere else in this thesis)")
    ap.add_argument("--fixed-tau-ms", type=float, default=FIXED_TAU_MS,
                    help="dye tau held fixed while camera dt is swept")
    ap.add_argument("--resume", default=None,
                    help="existing (possibly partial/interrupted) npz from a "
                         "previous run of this script -- already-computed "
                         "(kind, param) points are loaded and skipped rather "
                         "than recomputed. Point (kind, param) identity only, "
                         "not fixed-cam-ms/fixed-tau-ms -- only resume from a "
                         "run that used the same fixed values.")
    ap.add_argument("--density", type=float, default=DENSITY,
                    help="operating-point density for precision / recall / F1 / "
                         "recall_exc / recall_inh: the top `density` fraction of "
                         "|A| off-diagonal entries count as predicted edges "
                         "(default 0.10 ~ true Brunel epsilon)")
    ap.add_argument("--est-dir", default=None,
                    help="directory for the per-point OLS estimate A (float32 "
                         ".npy) + a _meta.npz (adj, n_exc), used by "
                         "fig_r2_rescore.py to add / retune a metric without "
                         "re-running this sweep. Default: "
                         "<out-without-.npz>_estimates/")
    ap.add_argument("--no-save-estimates", action="store_true",
                    help="skip the per-point A dump (~6 MB/point at N=1250)")
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
    n_exc = int(cfg["n_excitatory"])

    if args.no_save_estimates:
        est_dir = None
    elif args.est_dir:
        est_dir = Path(args.est_dir)
    else:
        stem = Path(args.out).with_suffix("")
        est_dir = stem.parent / f"{stem.name}_estimates"
    if est_dir is not None:
        est_dir.mkdir(parents=True, exist_ok=True)
        np.savez(est_dir / "_meta.npz", adj=adj.astype(np.float32),
                 n_exc=n_exc, N=N, net=args.net, density=args.density)
        print(f"per-point estimates -> {est_dir}", flush=True)

    def checkpoint(out):
        """Write whatever has been computed so far -- called after EVERY point,
        not just at the end, so a killed/interrupted job (this sweep takes
        ~1h and has twice been killed by the machine going unreachable
        mid-run) never loses more than the point in flight."""
        save = dict(net=args.net, N=N, T_ms=args.T_ms, density=args.density,
                    metrics=np.array(METRICS),
                    fixed_tau_ms=args.fixed_tau_ms, fixed_cam_ms=args.fixed_cam_ms)
        for k, v in out.items():
            if not v:
                continue
            arr = np.array(sorted(v))          # rows: (x, ratio, *METRICS)
            save[f"{k}_x"], save[f"{k}_ratio"] = arr[:, 0], arr[:, 1]
            for i, mn in enumerate(METRICS):
                save[f"{k}_{mn}"] = arr[:, 2 + i]
        tmp = f"{args.out}.tmp.npz"
        np.savez(tmp, **save)
        os.replace(tmp, args.out)   # atomic: never leaves a half-written file

    out = {k: [] for k in ["spikes", "deconv_tau", "deconv_rate", "raw_tau", "raw_rate"]}
    done = set()
    if args.resume:
        prev = np.load(args.resume, allow_pickle=False)
        for k in out:
            if f"{k}_x" not in prev.files:
                continue
            xs, rs = prev[f"{k}_x"], prev[f"{k}_ratio"]
            # tolerate a resume npz from before a metric was added: missing
            # column -> NaN, so the point still counts as done and is skipped.
            cols = [prev[f"{k}_{mn}"] if f"{k}_{mn}" in prev.files
                    else np.full(len(xs), np.nan) for mn in METRICS]
            for row in zip(xs, rs, *cols):
                out[k].append(tuple(float(v) for v in row))
                done.add((k, round(float(row[0]), 6)))
        print(f"resumed {len(done)} points from {args.resume}", flush=True)

    plan = ([("spikes", b) for b in args.spike_bins if "spikes" in args.kinds]
            + [("deconv_tau", t) for t in args.taus if "deconv_tau" in args.kinds]
            + [("raw_tau", t) for t in args.taus if "raw_tau" in args.kinds]
            + [("deconv_rate", f) for f in args.frames if "deconv_rate" in args.kinds]
            + [("raw_rate", f) for f in args.frames if "raw_rate" in args.kinds])
    plan = [(kind, p) for kind, p in plan if (kind, round(p, 6)) not in done]
    for kind, p in plan:
        x_native, x_ratio, mets, A = run(
            kind, p, idx, tms, N, dt, adj, args.T_ms, args.chunk, args.seed,
            n_exc, density=args.density,
            fixed_cam_ms=args.fixed_cam_ms, fixed_tau_ms=args.fixed_tau_ms)
        out[kind].append((x_native, x_ratio) + tuple(mets[mn] for mn in METRICS))
        if est_dir is not None:
            np.save(est_dir / f"{kind}_{p:g}.npy", A.astype(np.float32))
        print(f"  {kind:12s} param={p:<6g} x={x_native:.4g}  "
              f"(dt/tau={x_ratio:.4g})  AUC={mets['auc']:.3f} corr={mets['corr']:.3f} "
              f"P={mets['precision']:.3f} R={mets['recall']:.3f} "
              f"Rexc={mets['recall_exc']:.3f} Rinh={mets['recall_inh']:.3f}", flush=True)
        checkpoint(out)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
