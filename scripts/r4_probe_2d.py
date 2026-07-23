"""
2-D regime probe: can we hit a realistic rate AND textbook-AI irregularity at
the same time?

Sweeps (g, eta) — g raises irregularity, eta sets the rate — and reports firing
rate, ISI CV and synchrony for each combination. J is fluctuation-scaled per size
(J*sqrt(C_E)=0.1*sqrt(1000)); --j-scale multiplies it if g alone cannot lift CV.

Motivation: the R.4 ladder sits at CV ~0.63 because lowering eta for a realistic
~14 Hz also costs irregularity. Textbook AI is CV ~1. This probe finds whether
both targets are reachable together before we spend hours on the scaling sweeps.

Everything is saved: a CSV of every combination (for the record) and an npz for
the figure. Statistics come from sparse spike events; the dense (N,T) matrix is
never built.

RUN ON THE CLUSTER:
  sbatch slurm/run_r4_probe2d.slurm
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from calcium_ar.simulation.brunel_network import BrunelNetwork
from r4_tune_regime import sparse_stats, JBASE, SIZES

FIELDS = ["N", "N_E", "C_E", "J", "g", "eta", "rate", "cv", "sync", "silent", "score"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1250])
    ap.add_argument("--g", type=float, nargs="+", default=[5, 6, 7, 8, 10, 12])
    ap.add_argument("--eta", type=float, nargs="+",
                    default=[0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0])
    ap.add_argument("--j-scale", type=float, default=1.0,
                    help="multiply the fluctuation-scaled J (a 3rd knob for CV)")
    ap.add_argument("--sim-time", type=float, default=4000.0)
    ap.add_argument("--warmup-ms", type=float, default=1000.0)
    ap.add_argument("--n-threads", type=int, default=8)
    ap.add_argument("--target-rate", type=float, default=14.0)
    ap.add_argument("--target-cv", type=float, default=1.0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "results" / "regime2d")
    out_dir.mkdir(parents=True, exist_ok=True)

    for N, NE, NI in [s for s in SIZES if s[0] in args.sizes]:
        C_E = int(0.1 * NE)
        J = args.j_scale * JBASE / np.sqrt(C_E)
        print(f"\n=== N={N}  C_E={C_E}  J={J:.3f} (x{args.j_scale:g})  "
              f"targets: rate {args.target_rate} Hz, CV {args.target_cv} ===")
        print(f"{'g':>5} {'eta':>5} | {'rate':>6} {'CV':>5} {'sync':>6} {'silent':>6}")
        print("-" * 46)

        rate = np.full((len(args.eta), len(args.g)), np.nan)
        cv = np.full_like(rate, np.nan)
        sync = np.full_like(rate, np.nan)
        rows = []
        for gi, g in enumerate(args.g):
            for ei, eta in enumerate(args.eta):
                net = BrunelNetwork(n_excitatory=NE, n_inhibitory=NI, epsilon=0.1,
                                    g=g, eta=eta, J_ex=J, V_reset=10.0, delay=1.5,
                                    sim_time=args.sim_time, dt=0.1,
                                    n_threads=args.n_threads, seed=1)
                net.build(); net.run(densify=False)
                idx, tms = net.get_spike_events()
                r = sparse_stats(idx, tms, NE + NI, args.sim_time, args.warmup_ms)
                del net
                # distance from BOTH targets (relative), lower is better
                sc = (abs(r["rate"] - args.target_rate) / args.target_rate
                      + abs(r["cv"] - args.target_cv) / args.target_cv)
                rate[ei, gi], cv[ei, gi], sync[ei, gi] = r["rate"], r["cv"], r["sync"]
                rows.append(dict(N=N, N_E=NE, C_E=C_E, J=round(J, 4), g=g, eta=eta,
                                 rate=round(r["rate"], 2), cv=round(r["cv"], 3),
                                 sync=round(r["sync"], 4),
                                 silent=round(r["silent"], 3), score=round(sc, 4)))
                print(f"{g:>5g} {eta:>5.2f} | {r['rate']:>6.1f} {r['cv']:>5.2f} "
                      f"{r['sync']:>6.3f} {r['silent']:>6.2f}", flush=True)

        tag = f"N{N}_j{args.j_scale:g}"
        csv_path = out_dir / f"regime2d_{tag}.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
        np.savez(out_dir / f"regime2d_{tag}.npz", g=np.array(args.g),
                 eta=np.array(args.eta), rate=rate, cv=cv, sync=sync, N=N, J=J,
                 j_scale=args.j_scale, target_rate=args.target_rate,
                 target_cv=args.target_cv)

        best = min(rows, key=lambda d: d["score"])
        print(f"\n  best joint (rate~{args.target_rate}, CV~{args.target_cv}) at N={N}:")
        print(f"    g={best['g']:g}  eta={best['eta']:.2f}  J={best['J']:.3f}"
              f"  -> rate {best['rate']:.1f} Hz, CV {best['cv']:.2f}, "
              f"sync {best['sync']:.3f}")
        print(f"  wrote {csv_path} and .npz")

    print("\nplot with:  python scripts/fig_regime2d_plot.py "
          f"--data {out_dir}/regime2d_<tag>.npz")


if __name__ == "__main__":
    main()
