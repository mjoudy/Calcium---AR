"""
Network-state comparison of the two N=1250 regimes:

  n1250    g=5, eta=2   (old baseline: high-rate, more regular/correlated)
  n1250ai  g=8, eta=1   (regime-scan winner: clean asynchronous-irregular, low rate)

Side-by-side Brunel-style panel per regime: spike raster, population firing rate,
ISI distribution (with CV), and a stats box (E/I rates, CV, synchrony, silent).
This visualises why the clean-AI regime is harder to infer from: it is far less
correlated (low synchrony) and fires much less -> little dependency signal for a
regression method.

Usage:  python scripts/regime_compare.py [--sim-time 10000] [--out fig.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from calcium_ar.simulation.brunel_network import BrunelNetwork
from calcium_ar.preprocessing.signal_utils import calculate_cv
from scripts.regime_scan import synchrony_index

N_E, N_I = 1000, 250
REGIMES = [("n1250  (g=5, eta=2)", dict(g=5.0, eta=2.0, J_ex=0.8), "#e34948"),
           ("n1250ai  (g=8, eta=1)", dict(g=8.0, eta=1.0, J_ex=0.8), "#2a78d6")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
C_E, C_I = "#2a78d6", "#e34948"


def simulate(cfg, sim_time, dt, seed=1):
    net = BrunelNetwork(n_excitatory=N_E, n_inhibitory=N_I, epsilon=0.1,
                        g=cfg["g"], eta=cfg["eta"], J_ex=cfg["J_ex"], delay=1.5,
                        sim_time=sim_time, dt=dt, n_threads=4, seed=seed)
    net.build(); net.run()
    spikes, _ = net.get_results()
    return spikes


def stats(spikes, dt, warmup_ms=500.0):
    w = int(round(warmup_ms / dt))
    spk = spikes[:, w:]
    T_sec = spk.shape[1] * dt / 1000.0
    rates = spk.sum(1) / T_sec
    cv = calculate_cv(spk)
    cv_mean = float(np.nanmean(np.asarray(cv)[rates > 0]))
    return dict(rate_E=float(rates[:N_E].mean()), rate_I=float(rates[N_E:].mean()),
                rate_all=float(rates.mean()), cv=cv_mean,
                sync=synchrony_index(spk, dt), silent=float((rates == 0).mean()),
                rates=rates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-time", type=float, default=10000.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--window-ms", type=float, default=1000.0, help="raster/rate window")
    ap.add_argument("--out", default=str(ROOT / "results" / "regime_compare.png"))
    args = ap.parse_args()

    dt = args.dt
    win = int(round(args.window_ms / dt))
    t_off = int(round(1000.0 / dt))                     # start window after 1 s
    n_show = 120                                        # neurons in raster (E + I)
    ei_idx = np.r_[np.arange(0, n_show * N_E // (N_E + N_I)),
                   np.arange(N_E, N_E + n_show * N_I // (N_E + N_I))]

    fig, axes = plt.subplots(3, 2, figsize=(13, 10),
                             gridspec_kw=dict(height_ratios=[1.4, 0.9, 0.9]))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.08, hspace=0.45, wspace=0.2)

    for col, (name, cfg, accent) in enumerate(REGIMES):
        spikes = simulate(cfg, args.sim_time, dt)
        st = stats(spikes, dt)
        tsl = slice(t_off, t_off + win)
        tvec = np.arange(win) * dt

        # raster
        axr = axes[0, col]
        for row, n in enumerate(ei_idx):
            ts = np.where(spikes[n, tsl] > 0)[0] * dt
            c = C_E if n < N_E else C_I
            axr.plot(ts, np.full_like(ts, row), "|", color=c, ms=4, mew=0.8)
        axr.set(xlim=(0, args.window_ms), ylim=(-1, len(ei_idx)),
                xlabel="time (ms)", ylabel="neuron", title=name)
        axr.set_yticks([])

        # population rate (Hz), 5 ms bins
        axp = axes[1, col]
        b = int(round(5.0 / dt))
        pop = spikes[:, tsl].reshape(spikes.shape[0], -1)[:, :(win // b) * b]
        pr = pop.reshape(spikes.shape[0], win // b, b).sum(2).sum(0) / (spikes.shape[0] * b * dt / 1000.0)
        axp.plot(np.arange(len(pr)) * 5.0, pr, color=accent, lw=1.3)
        axp.set(xlim=(0, args.window_ms), xlabel="time (ms)", ylabel="pop. rate (Hz)")
        axp.grid(True, color=GRID, lw=0.6); axp.set_axisbelow(True)

        # ISI distribution (pooled) + stats box
        axi = axes[2, col]
        isis = []
        for n in range(spikes.shape[0]):
            ts = np.where(spikes[n] > 0)[0] * dt
            if len(ts) > 1:
                isis.append(np.diff(ts))
        isis = np.concatenate(isis) if isis else np.array([1.0])
        axi.hist(isis, bins=np.linspace(0, np.percentile(isis, 99), 60),
                 color=accent, alpha=0.8, density=True)
        axi.set(xlabel="ISI (ms)", ylabel="density")
        axi.grid(True, color=GRID, lw=0.6); axi.set_axisbelow(True)
        txt = (f"rate:  E {st['rate_E']:.1f} Hz   I {st['rate_I']:.1f} Hz\n"
               f"mean rate {st['rate_all']:.1f} Hz\n"
               f"CV(ISI) {st['cv']:.2f}   synchrony {st['sync']:.3f}\n"
               f"silent {100*st['silent']:.0f}%")
        axi.text(0.97, 0.95, txt, transform=axi.transAxes, ha="right", va="top",
                 fontsize=9, color=INK,
                 bbox=dict(boxstyle="round", fc="white", ec=accent, lw=1.3))

    fig.suptitle("Network-state comparison, N=1250  —  old baseline vs clean-AI regime",
                 fontsize=14, color=INK, x=0.07, ha="left")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
