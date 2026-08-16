"""
Turn saved Hawkes spike events (scripts/hawkes_ground_truth.py) into Cxx/Cyx
moments via the SAME calcium -> deconvolution -> streaming-moments pipeline
used for the LIF/PIF arms (calcium_ar.experiments.streaming.stream_moments,
also used by scripts/wrapup_run_stream.py) -- same tau/amplitude/noise/smooth-
window/lag as the project's validated landscape (COMMON dict in
scripts/wrapup_run.py).

This is the piece the OU arm skipped entirely (OU's Cxx/Cyx come straight from
a closed-form Lyapunov solve, no spikes, no calcium, no deconvolution). Running
the Hawkes arm through the real observation pipeline separates two previously
conflated questions: is the false-positive confound suppressed because the
DYNAMICS are linear, or because the OU arm also skipped imaging/deconvolution?
Hawkes-with-calcium answers "linear dynamics, real pipeline"; the existing OU
arm remains "linear dynamics, no pipeline at all" for contrast.

Output format matches ou_linear_ground_truth.py / best_moments/* exactly
(Cxx.npy, Cyx.npy, adj_true.npy) -- drop straight into fig_linearity_way2.py's
DEFAULT_SOURCES.

Usage:
  python scripts/hawkes_to_moments.py \\
      --events-dir ~/calcium_results/hawkes_events/n1250 \\
      --out ~/calcium_results/hawkes_moments/n1250_calcium
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calcium_ar.experiments.streaming import stream_moments
from calcium_ar.solvers.from_moments import ols_from_moments

# Matches COMMON in scripts/wrapup_run.py -- keep in sync, this IS the
# apples-to-apples point of the comparison.
DEFAULTS = dict(tau=100.0, amplitude=1.0, sigma_intra=0.01, sigma_extra=0.05,
                smooth_window_ms=3.1, tau_method="ransac", lag_ms=2.0, dt=0.1,
                chunk_ms=10_000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=DEFAULTS["tau"])
    ap.add_argument("--amplitude", type=float, default=DEFAULTS["amplitude"])
    ap.add_argument("--sigma-intra", type=float, default=DEFAULTS["sigma_intra"])
    ap.add_argument("--sigma-extra", type=float, default=DEFAULTS["sigma_extra"])
    ap.add_argument("--smooth-window-ms", type=float, default=DEFAULTS["smooth_window_ms"])
    ap.add_argument("--tau-method", default=DEFAULTS["tau_method"])
    ap.add_argument("--lag-ms", type=float, default=DEFAULTS["lag_ms"])
    ap.add_argument("--dt", type=float, default=DEFAULTS["dt"])
    ap.add_argument("--chunk-ms", type=float, default=DEFAULTS["chunk_ms"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--density", type=float, default=0.10, help="for the quick OLS sanity check")
    ap.add_argument("--known-tau", action="store_true",
                    help="skip RANSAC estimation, use --tau directly for deconvolution -- "
                         "valid here because this is synthetic data with a KNOWN true "
                         "calcium decay (we generated it), unlike real/LIF data. RANSAC "
                         "kept landing far from the true value unpredictably on Hawkes "
                         "signals (-330/+148/+54/-1286ms across similar runs).")
    args = ap.parse_args()

    d = Path(args.events_dir).expanduser()
    idx = np.load(d / "idx.npy").astype(np.int32)
    times_ms = np.load(d / "times_ms.npy").astype(np.float64)
    adj_true = np.load(d / "adj_true.npy").astype(np.float64)
    meta = np.load(d / "meta.npy", allow_pickle=True).item()
    N = adj_true.shape[0]
    sim_time = float(meta["sim_time"])
    print(f"loaded {len(idx)} events, N={N}, sim_time={sim_time:.0f}ms  (from {d})", flush=True)

    lag = max(1, round(args.lag_ms / args.dt))
    w = max(5, round(args.smooth_window_ms / args.dt))
    smooth_win = w if w % 2 == 1 else w + 1
    checkpoint = int(round(sim_time / args.dt))
    chunk_samples = int(round(args.chunk_ms / args.dt))

    print("streaming calcium -> deconvolution -> moments (same pipeline as LIF/PIF) ...",
          flush=True)
    moments, tau_est, rate = stream_moments(
        net=None, N=N, sim_time=sim_time, spike_events=(idx, times_ms), dt=args.dt,
        lag=lag, tau=args.tau, amplitude=args.amplitude,
        sigma_intra=args.sigma_intra, sigma_extra=args.sigma_extra,
        smooth_win=smooth_win, tau_method=args.tau_method,
        checkpoints_samples=[checkpoint], chunk_samples=chunk_samples, seed=args.seed,
        known_tau=(args.tau if args.known_tau else None),
    )
    Cxx, Cyx = moments[checkpoint]
    tau_mean = float(np.mean(tau_est))
    tau_str = (f"{tau_mean:.1f}ms (mean over {np.size(tau_est)} neurons)"
               if np.ndim(tau_est) else f"{tau_mean:.1f}ms")
    print(f"moments built. mean rate {rate:.2f} Hz, estimated calcium tau {tau_str}",
          flush=True)
    # The tau estimate is fit ONCE, from the first chunk only (see
    # stream_moments) -- if that chunk is too short (--chunk-ms too small) or
    # otherwise unrepresentative, the RANSAC fit can come out negative or wildly
    # off, silently feeding a broken deconvolution into everything downstream.
    # It should land near --tau (the value used to GENERATE the calcium signal
    # in the first place); flag loudly rather than let a bad run look normal.
    if tau_mean <= 0 or not (0.3 * args.tau <= tau_mean <= 3.0 * args.tau):
        print(f"WARNING: estimated tau ({tau_mean:.1f}ms) is far from the true "
              f"generative tau ({args.tau:.1f}ms) -- the deconvolution step likely "
              f"broke (e.g. --chunk-ms too small for the one-time tau fit to have "
              f"enough data). Treat Cxx/Cyx/downstream A as UNRELIABLE until "
              f"rerun with a larger --chunk-ms.", flush=True)

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "Cxx.npy", Cxx)
    np.save(out / "Cyx.npy", Cyx)
    np.save(out / "adj_true.npy", adj_true)
    print(f"wrote {out}/Cxx.npy Cyx.npy adj_true.npy", flush=True)

    # quick sanity check, same style as ou_linear_ground_truth.py
    A = ols_from_moments(Cxx, Cyx)
    aa = np.abs(A)
    off = ~np.eye(N, dtype=bool)
    truth = (adj_true.T != 0)
    tau_thr = np.quantile(aa[off], 1.0 - args.density)
    pred = aa > tau_thr
    tp = (pred & truth & off).sum(); fp = (pred & ~truth & off).sum()
    prec = tp / max(tp + fp, 1)
    print(f"sanity check (top-{args.density * 100:.0f}% |A_ols|): precision={prec:.3f}  "
          f"mean|A| true edges={aa[truth & off].mean():.4g}  "
          f"mean|A| non-edges={aa[~truth & off].mean():.4g}")


if __name__ == "__main__":
    main()
