"""
Compare ground-truth network configurations at full Brunel scale (N=12500).

Motivation: our n12500ai config was NOT Brunel's. We tuned (g, eta) empirically at
N=1250 and then fluctuation-scaled J from *our* value (0.8 -> 0.23) rather than
using the canonical J=0.1 that applies directly at this size. Separately, this
codebase used V_reset = 0 mV while Brunel 2000 uses V_r = 10 mV — which changes
the f-I curve and therefore shifts the whole (g, eta) phase diagram.

This probe runs a short simulation of each configuration and reports the regime
diagnostics (firing rate, ISI CV, synchrony), so we can see:
  1. what regime the CANONICAL Brunel parameters actually produce here, and
  2. how much of our deviation is explained by V_reset alone.

Regime-only (no calcium, no inference) so it is cheap.

Usage:  python scripts/regime_probe.py [--sim-time 10000] [--n-threads 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from calcium_ar.simulation.brunel_network import BrunelNetwork
from calcium_ar.preprocessing.signal_utils import calculate_cv
from scripts.regime_scan import synchrony_index

# Brunel 2000 canonical: N_E=10000, N_I=2500, eps=0.1 (C_E=1250), J=0.1 mV,
# delay 1.5 ms, tau_m 20, theta 20, V_r 10, t_ref 2.
#
# Two scales:
#   n12500 — canonical Brunel size, so J=0.1 applies directly (no scaling needed).
#   n1250  — cheap V_reset isolation against configs whose behaviour we already
#            know: (g=5,eta=2,J=0.8) gave 42 Hz / CV 0.58 and (g=8,eta=1,J=0.8)
#            gave 8 Hz / CV 0.82, both at V_reset=0.
CONFIGS_BY_SCALE = {
    "n12500": [
        ("brunel_canonical",     dict(J_ex=0.1,  g=5.0, eta=2.0, V_reset=10.0)),
        ("brunel_canonical_Vr0", dict(J_ex=0.1,  g=5.0, eta=2.0, V_reset=0.0)),
        ("ours_n12500ai",        dict(J_ex=0.23, g=8.0, eta=1.0, V_reset=0.0)),
        ("ours_params_Vr10",     dict(J_ex=0.23, g=8.0, eta=1.0, V_reset=10.0)),
    ],
    "n1250": [
        ("old_g5eta2_Vr0",       dict(J_ex=0.8, g=5.0, eta=2.0, V_reset=0.0)),
        ("old_g5eta2_Vr10",      dict(J_ex=0.8, g=5.0, eta=2.0, V_reset=10.0)),
        ("ai_g8eta1_Vr0",        dict(J_ex=0.8, g=8.0, eta=1.0, V_reset=0.0)),
        ("ai_g8eta1_Vr10",       dict(J_ex=0.8, g=8.0, eta=1.0, V_reset=10.0)),
    ],
}


def probe(cfg, n_exc, n_inh, sim_time, dt, warmup_ms, n_threads, seed=1):
    net = BrunelNetwork(n_excitatory=n_exc, n_inhibitory=n_inh, epsilon=0.1,
                        g=cfg["g"], eta=cfg["eta"], J_ex=cfg["J_ex"],
                        V_reset=cfg["V_reset"], delay=1.5,
                        sim_time=sim_time, dt=dt, n_threads=n_threads, seed=seed)
    net.build(); net.run()
    spikes, _ = net.get_results()
    w = int(round(warmup_ms / dt))
    spk = spikes[:, w:]
    rates = spk.sum(1) / (spk.shape[1] * dt / 1000.0)
    cv = np.asarray(calculate_cv(spk))
    cv_mean = float(np.nanmean(cv[rates > 0])) if (rates > 0).any() else float("nan")
    return dict(rate_E=float(rates[:n_exc].mean()), rate_I=float(rates[n_exc:].mean()),
                rate=float(rates.mean()), cv=cv_mean,
                sync=synchrony_index(spk, dt), silent=float((rates == 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="n12500", choices=list(CONFIGS_BY_SCALE))
    ap.add_argument("--n-exc", type=int, default=None)
    ap.add_argument("--n-inh", type=int, default=None)
    ap.add_argument("--sim-time", type=float, default=10000.0)
    ap.add_argument("--warmup-ms", type=float, default=1000.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--n-threads", type=int, default=8)
    args = ap.parse_args()

    defaults = {"n12500": (10000, 2500), "n1250": (1000, 250)}[args.scale]
    args.n_exc = args.n_exc or defaults[0]
    args.n_inh = args.n_inh or defaults[1]
    CONFIGS = CONFIGS_BY_SCALE[args.scale]

    print(f"scale={args.scale}  N={args.n_exc + args.n_inh}  "
          f"(C_E={int(0.1*args.n_exc)})  sim_time={args.sim_time:.0f} ms\n")
    print(f"{'config':22s} {'J':>6} {'g':>4} {'eta':>5} {'V_r':>5} | "
          f"{'rate':>6} {'rateE':>6} {'rateI':>6} {'CV':>5} {'sync':>6} {'silent':>6}")
    print("-" * 96)
    for name, cfg in CONFIGS:
        r = probe(cfg, args.n_exc, args.n_inh, args.sim_time, args.dt,
                  args.warmup_ms, args.n_threads)
        print(f"{name:22s} {cfg['J_ex']:6.2f} {cfg['g']:4.0f} {cfg['eta']:5.1f} "
              f"{cfg['V_reset']:5.1f} | {r['rate']:6.1f} {r['rate_E']:6.1f} "
              f"{r['rate_I']:6.1f} {r['cv']:5.2f} {r['sync']:6.3f} {r['silent']:6.2f}",
              flush=True)
    print("\nAI target: rate ~5-15 Hz, CV ~0.8-1.0, sync ~0")


if __name__ == "__main__":
    main()
