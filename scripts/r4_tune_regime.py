"""
R.4 regime tuning: hold the SAME regime (AI) and the SAME rate across network
sizes, so the scaling sweep varies only N.

g=6, V_reset=10, J fluctuation-scaled (J*sqrt(C_E)=const=0.1*sqrt(1000)), and eta
tuned per size to hit a target rate (default ~14 Hz, matching the existing
N=12500 low-rate run so that run can be reused).

Memory note: statistics are computed from the SPARSE spike events — the dense
(N, T) matrix is never built. Densifying at N=12500 costs ~4 GB per simulation,
which is what makes a naive probe unrunnable on a laptop.

RUN THIS ON THE CLUSTER (it is ~30 short simulations, including several at
N=12500):
  srun --cpus-per-task=8 --mem=16G --time=00:30:00 \
    $(ws_find calcium_ar)/env/bin/python scripts/r4_tune_regime.py --target 14
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from calcium_ar.simulation.brunel_network import BrunelNetwork

JBASE = 0.1 * np.sqrt(1000.0)          # fluctuation invariant J*sqrt(C_E)

# (N, N_E, N_I) — Brunel 4:1 E:I ratio
SIZES = [(1250, 1000, 250), (2500, 2000, 500),
         (5000, 4000, 1000), (12500, 10000, 2500)]
# Small networks fire FASTER under fluctuation-scaling, so they need a lower eta
# to reach the same rate: the grid must reach well below 1 for N=1250.
ETAS = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.75, 2.0]


def sparse_stats(idx, times, N, T_ms, warmup_ms, bin_ms=2.0, n_sub=200, seed=0):
    """Rate / ISI-CV / synchrony from sparse events only (no dense (N,T))."""
    keep = times >= warmup_ms
    idx, times = idx[keep], times[keep] - warmup_ms
    dur_s = (T_ms - warmup_ms) / 1000.0
    rate = len(times) / (N * dur_s)

    order = np.lexsort((times, idx))                 # group by neuron, then time
    i_s, t_s = idx[order], times[order]
    starts = np.flatnonzero(np.r_[True, np.diff(i_s) != 0])
    ends = np.r_[starts[1:], len(i_s)]
    cvs = []
    for s, e in zip(starts, ends):
        if e - s > 2:
            isi = np.diff(t_s[s:e])
            mu = isi.mean()
            if mu > 0:
                cvs.append(isi.std() / mu)
    cv = float(np.mean(cvs)) if cvs else float("nan")
    silent = 1.0 - len(starts) / N

    rng = np.random.default_rng(seed)                # synchrony on a subset
    sub = rng.choice(N, size=min(n_sub, N), replace=False)
    pos = -np.ones(N, dtype=np.int64); pos[sub] = np.arange(len(sub))
    nb = max(2, int(dur_s * 1000.0 / bin_ms))
    counts = np.zeros((len(sub), nb))
    sel = pos[idx] >= 0
    b = np.clip((times[sel] / bin_ms).astype(np.int64), 0, nb - 1)
    np.add.at(counts, (pos[idx[sel]], b), 1.0)
    active = counts.sum(1) > 0
    if active.sum() > 2:
        C = np.corrcoef(counts[active])
        off = ~np.eye(C.shape[0], dtype=bool)
        sync = float(np.nanmean(C[off]))
    else:
        sync = float("nan")
    return dict(rate=rate, cv=cv, sync=sync, silent=silent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=14.0, help="target rate (Hz)")
    ap.add_argument("--sim-time", type=float, default=4000.0)
    ap.add_argument("--warmup-ms", type=float, default=1000.0)
    ap.add_argument("--n-threads", type=int, default=8)
    ap.add_argument("--sizes", type=int, nargs="+", default=None,
                    help="subset of N to tune (default: all four)")
    args = ap.parse_args()

    sizes = [s for s in SIZES if args.sizes is None or s[0] in args.sizes]
    print(f"target rate ~{args.target} Hz   g=6  V_reset=10  "
          f"J*sqrt(C_E)={JBASE:.3f}\n")
    print(f"{'N':>6} {'C_E':>5} {'J':>6} {'eta':>5} | {'rate':>6} {'CV':>5} "
          f"{'sync':>6} {'silent':>6}")
    print("-" * 60)
    picks = {}
    for N, NE, NI in sizes:
        C_E = int(0.1 * NE)
        J = JBASE / np.sqrt(C_E)
        best = None
        for eta in ETAS:
            net = BrunelNetwork(n_excitatory=NE, n_inhibitory=NI, epsilon=0.1,
                                g=6.0, eta=eta, J_ex=J, V_reset=10.0, delay=1.5,
                                sim_time=args.sim_time, dt=0.1,
                                n_threads=args.n_threads, seed=1)
            net.build(); net.run(densify=False)       # <- never densify
            idx, tms = net.get_spike_events()
            r = sparse_stats(idx, tms, NE + NI, args.sim_time, args.warmup_ms)
            del net
            print(f"{N:>6} {C_E:>5} {J:>6.3f} {eta:>5.2f} | {r['rate']:>6.1f} "
                  f"{r['cv']:>5.2f} {r['sync']:>6.3f} {r['silent']:>6.2f}", flush=True)
            if best is None or abs(r["rate"] - args.target) < abs(best[1] - args.target):
                best = (eta, r["rate"], r["cv"], r["sync"], J)
        picks[N] = best
        print(f"  -> N={N}: eta={best[0]:.2f}  rate={best[1]:.1f} Hz  "
              f"CV={best[2]:.2f}  sync={best[3]:.3f}  J={best[4]:.3f}\n", flush=True)

    print("=== R.4 configs ===")
    for N, (eta, rate, cv, sync, J) in picks.items():
        NE = [s[1] for s in SIZES if s[0] == N][0]
        print(f"  N={N:>6}: NE={NE:>5}  J={J:.3f}  eta={eta:.2f}   "
              f"(rate {rate:.1f} Hz, CV {cv:.2f}, sync {sync:.3f})")


if __name__ == "__main__":
    main()
