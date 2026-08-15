"""
Eta probe for the PIF-pilot tau_m ladder (professor's comment on
fig_linearity_way2: "you could make it even larger, factor 10000 for
example"). Same purpose as scripts/r4_tune_regime.py's N sweep, but the knob
here is tau_m, not N: nu_thr scales with tau_m (Brunel mean-field: nu_thr =
theta/(J*C_E*tau_m)), so raising tau_m without re-tuning eta collapses the
rate. The existing tau_m x10 pilot (20ms -> 200ms) needed eta 1.30 -> 8.0 by
DIRECT PROBE, not a closed-form scaling (naive nu_thr~1/tau_m predicts x10,
the measured ratio was x6.15 -- the network is already far from mean-field at
CV~1.9) -- so each further step needs its own probe too, not a guess.

CAUTION -- read before running the x1000/x10000 stages: tau_m=20000ms (20s) or
200000ms (200s) means the membrane time constant approaches or EXCEEDS a short
probe's sim_time. A network can't be assumed to reach a stationary firing
regime within a duration much shorter than its own integration time constant.
This script warns (does not block) when --sim-time < 20*tau_m; treat a
triggered warning as "these numbers may not mean anything yet," and rerun with
a longer sim_time (which pushes this from a probe into serious compute -- do
that on the cluster, and expect it to cost much more than the earlier stages,
possibly needing a longer PRODUCTION sim_time too, not just a longer probe).

RUN ON THE CLUSTER (short sims, but many of them -- same reasoning as
r4_tune_regime.py; sparse events only, dense (N,T) never built):
  srun --cpus-per-task=8 --mem=16G --time=00:30:00 \\
    $(ws_find calcium_ar)/env/bin/python scripts/pif_tau_probe.py --stage x100

Usage:
  python scripts/pif_tau_probe.py --stage x100    # tau_m=2000ms   (n1250_pif x10)
  python scripts/pif_tau_probe.py --stage x1000   # tau_m=20000ms
  python scripts/pif_tau_probe.py --stage x10000  # tau_m=200000ms -- read the caution above first
  python scripts/pif_tau_probe.py --tau-m 2000 --etas 20 30 45 65 90 120 160 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from calcium_ar.simulation.brunel_network import BrunelNetwork
from r4_tune_regime import sparse_stats  # reuse the exact rate/CV/sync/silent stats

# n1250_r4 / n1250_pif's fixed parameters (see NETS in scripts/wrapup_run.py) --
# only tau_m and eta change here.
NE, NI = 1000, 250
J_EX, G, V_RESET, EPSILON = 0.474, 8.0, 10.0, 0.1
BASE_TAU_M = 20.0   # ms, Brunel default
PIF_TAU_M = 200.0   # ms, the existing x10 pilot (eta=8.0 -> 13.7 Hz, CV 1.94)
PIF_ETA = 8.0

# Coarse brackets around a naive nu_thr~1/tau_m scaling, corrected down by the
# ~0.6x factor observed going from x1 -> x10 (see module docstring) -- wide on
# purpose since the correction factor itself is not known to hold further out.
STAGES = {
    # x100 grid NARROWED 2026-08-15 from a first coarse pass (etas 20-200,
    # sim_time=4000ms, which triggered the sim_time<20*tau_m warning below --
    # its eta=90 pick is NOT trusted): target 14Hz bracketed between eta=65
    # (12.0Hz) and eta=90 (15.7Hz). Re-centered here, now run at a trustworthy
    # sim_time (auto-scaled by --sim-time's default, see main()).
    "x100":   dict(tau_m=BASE_TAU_M * 100,   etas=[60, 68, 75, 80, 85, 92, 100, 110]),
    "x1000":  dict(tau_m=BASE_TAU_M * 1000,  etas=[150, 250, 400, 600, 900, 1300, 1800, 2500]),
    "x10000": dict(tau_m=BASE_TAU_M * 10000, etas=[800, 1500, 2500, 4000, 6000, 9000, 13000, 18000]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES), default=None,
                    help="preset tau_m + eta grid; overrides --tau-m/--etas")
    ap.add_argument("--tau-m", type=float, default=None, help="ms")
    ap.add_argument("--etas", type=float, nargs="+", default=None)
    ap.add_argument("--target", type=float, default=14.0, help="target rate (Hz), matches n1250_r4")
    ap.add_argument("--sim-time", type=float, default=None,
                    help="ms; default auto-scales to 20x tau_m (the script's own "
                         "stationarity floor) unless given explicitly")
    ap.add_argument("--warmup-ms", type=float, default=None,
                    help="ms; default auto-scales to 5x tau_m unless given explicitly")
    ap.add_argument("--n-threads", type=int, default=8)
    args = ap.parse_args()

    if args.stage:
        tau_m = STAGES[args.stage]["tau_m"]
        etas = args.etas or STAGES[args.stage]["etas"]
    else:
        if args.tau_m is None or args.etas is None:
            raise SystemExit("pass --stage, or both --tau-m and --etas")
        tau_m, etas = args.tau_m, args.etas

    # Auto-scale so the stationarity warning below doesn't just keep firing at
    # every stage -- 20x/5x are the same multiples the warning itself checks.
    sim_time = args.sim_time if args.sim_time is not None else max(4000.0, 20 * tau_m)
    warmup_ms = args.warmup_ms if args.warmup_ms is not None else max(1000.0, 5 * tau_m)
    args.sim_time, args.warmup_ms = sim_time, warmup_ms

    # Rough cost heads-up before committing: this run's own eta=90 point did
    # 4000ms in 0.2s wall (8 threads) -- extrapolate linearly, serial over the
    # eta grid. Very approximate; NEST startup overhead means small sim_times
    # are relatively more expensive than this suggests.
    est_wall_s = len(etas) * sim_time * (0.2 / 4000.0)
    print(f"{len(etas)} eta points x sim_time={sim_time:.0f}ms  "
          f"(rough serial estimate: ~{est_wall_s:.0f}s wall, extrapolated from "
          f"today's x100 probe -- treat as order-of-magnitude only)\n", flush=True)

    if args.sim_time < 20 * tau_m:
        print(f"WARNING: sim_time={args.sim_time:.0f}ms is only "
              f"{args.sim_time / tau_m:.1f}x tau_m={tau_m:.0f}ms. A network may not "
              f"reach a stationary firing regime this fast when tau_m is this large "
              f"-- treat results below as provisional, not a real eta pick, until "
              f"rerun with a longer sim_time. See the module docstring.\n", flush=True)

    print(f"tau_m={tau_m:.0f}ms ({tau_m / BASE_TAU_M:.0f}x base)  target rate~{args.target}Hz  "
          f"J={J_EX}  g={G}  V_reset={V_RESET}  sim_time={args.sim_time:.0f}ms\n", flush=True)
    print(f"{'eta':>8} | {'rate':>6} {'CV':>5} {'sync':>6} {'silent':>6}")
    print("-" * 44)
    best = None
    for eta in etas:
        net = BrunelNetwork(n_excitatory=NE, n_inhibitory=NI, epsilon=EPSILON,
                            g=G, eta=eta, J_ex=J_EX, V_reset=V_RESET, delay=1.5,
                            tau_m=tau_m, sim_time=args.sim_time, dt=0.1,
                            n_threads=args.n_threads, seed=1)
        net.build(); net.run(densify=False)
        idx, tms = net.get_spike_events()
        r = sparse_stats(idx, tms, NE + NI, args.sim_time, args.warmup_ms)
        del net
        print(f"{eta:>8.2f} | {r['rate']:>6.1f} {r['cv']:>5.2f} {r['sync']:>6.3f} "
              f"{r['silent']:>6.2f}", flush=True)
        if best is None or abs(r["rate"] - args.target) < abs(best[1] - args.target):
            best = (eta, r["rate"], r["cv"], r["sync"])

    print(f"\n-> tau_m={tau_m:.0f}ms: closest eta={best[0]:.2f}  rate={best[1]:.1f}Hz  "
          f"CV={best[2]:.2f}  sync={best[3]:.3f}")
    print("If the closest grid point is at either end of the --etas grid, the true "
          "optimum is outside it -- rerun with a shifted grid before trusting this pick.")


if __name__ == "__main__":
    main()
