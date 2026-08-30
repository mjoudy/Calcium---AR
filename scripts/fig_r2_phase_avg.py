"""
R.2 follow-up: does averaging over camera START PHASE recover any of the
performance lost at coarse camera frame intervals?

Idea (see docs/experiments/notebook.md): a camera at a given frame interval
only ever keeps every r-th fine-resolution sample, always starting at the
same phase (t=0) in the existing R.2 figures. The "dropped" in-between
samples are equally valid recordings of the SAME stationary process, just
shifted a few ms later. This script reruns each coarse interval at several
different starting phases, pools their moments (raw-sum, exact for equal n
-- weighted by n when not exactly equal), solves ONCE per interval, and
compares that "phase-averaged" curve against the original single-phase
(phase=0) curve.

This does NOT add new independent data (all phases come from one
simulation) -- it only removes phase-selection noise. See the notebook
entry for the full reasoning.

Usage:
  python scripts/fig_r2_phase_avg.py --net n1250_r4 --T-ms 500000 \
      --intervals 10 20 33 50 100 200 500 1000 --n-phases 12 \
      --out figures/fig_R2_phase_avg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from calcium_ar.simulation.brunel_network import BrunelNetwork
from calcium_ar.experiments.streaming import MomentAccumulator
from calcium_ar.solvers.from_moments import ols_from_moments
from wrapup_run import build_cfg
from fig_r2_compute import iter_calcium, FIXED_TAU_MS, LAG_MS
from r2_metrics import metrics_from_A
import figstyle as fs


def accumulate_raw(feed_iter, N, lag):
    """Like fig_r2_compute.accumulate(), but returns the accumulator itself
    (not the normalized snapshot) so several phases' raw moments can be
    pooled exactly before the one final normalization."""
    acc = MomentAccumulator(N, lag)
    for fc in feed_iter:
        acc.add(np.asarray(fc, dtype=np.float64))
    return acc


def pooled_snapshot(accs):
    """Exact pooled (Cxx, Cyx) from several accumulators -- weighted by each
    one's own sample count n, not assumed equal (chunk-boundary rounding can
    make them differ by a handful of samples)."""
    n_total = sum(a.n for a in accs)
    Cxx_raw = sum(a.Cxx_raw for a in accs)
    Cyx_raw = sum(a.Cyx_raw for a in accs)
    s_prev = sum(a.s_prev for a in accs)
    s_now = sum(a.s_now for a in accs)
    mu_p, mu_n = s_prev / n_total, s_now / n_total
    Cxx = (Cxx_raw - n_total * np.outer(mu_p, mu_p)) / n_total
    Cyx = (Cyx_raw - n_total * np.outer(mu_n, mu_p)) / n_total
    return Cxx, Cyx


def one_point(kind, interval_ms, phases, idx, tms, N, dt, adj, T_ms, chunk,
              seed, n_exc, density):
    deconv = kind == "deconv"
    lag = max(1, round(LAG_MS / interval_ms))
    accs = []
    for ph in phases:
        rng = np.random.default_rng(seed)      # same noise draw every phase
        feed = iter_calcium(idx, tms, N, dt, FIXED_TAU_MS, T_ms, deconv,
                            interval_ms, chunk, rng, phase=ph)
        accs.append(accumulate_raw(feed, N, lag))
    single = accs[0].snapshot()                 # phase=0 alone == today's figure
    pooled = pooled_snapshot(accs)
    A_single = ols_from_moments(*single)
    A_pooled = ols_from_moments(*pooled)
    m_single = metrics_from_A(A_single, adj, n_exc, density)
    m_pooled = metrics_from_A(A_pooled, adj, n_exc, density)
    n_kept = [a.n for a in accs]
    return m_single, m_pooled, n_kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="n1250_r4")
    ap.add_argument("--T-ms", type=float, default=500_000.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--intervals", type=float, nargs="+",
                    default=[10, 20, 33, 50, 100, 200, 500, 1000])
    ap.add_argument("--n-phases", type=int, default=12,
                    help="phase offsets per interval, evenly spread over [0, r)")
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--out", default="figures/fig_R2_phase_avg")
    args = ap.parse_args()

    cfg = build_cfg(args.net); dt = cfg["dt"]
    print(f"simulating {args.net} for {args.T_ms:.0f} ms ...", flush=True)
    net = BrunelNetwork(n_excitatory=cfg["n_excitatory"], n_inhibitory=cfg["n_inhibitory"],
                        epsilon=cfg["epsilon"], g=cfg["g"], eta=cfg["eta"], J_ex=cfg["J_ex"],
                        delay=cfg["delay"], V_reset=cfg["V_reset"], sim_time=args.T_ms,
                        dt=dt, n_threads=cfg["n_threads"], seed=args.seed)
    net.build(); net.run(densify=False)
    idx, tms = net.get_spike_events(); idx = idx.astype(np.int64)
    adj = net.get_adjacency(); np.fill_diagonal(adj, 0.0)
    N = adj.shape[0]; n_exc = int(cfg["n_excitatory"])

    rows = []
    for interval in args.intervals:
        r = max(1, int(round(interval / dt)))
        n_ph = min(args.n_phases, r)
        phases = np.linspace(0, r, n_ph, endpoint=False, dtype=int).tolist()
        for kind in ("deconv", "raw"):
            m_single, m_pooled, n_kept = one_point(
                kind, interval, phases, idx, tms, N, dt, adj, args.T_ms,
                args.chunk, args.seed, n_exc, args.density)
            rows.append((interval, kind, m_single, m_pooled))
            print(f"interval={interval:<6g} {kind:7s} n_phases={n_ph:2d} "
                  f"n_kept~{np.mean(n_kept):.0f}  "
                  f"AUC single={m_single['auc']:.3f} pooled={m_pooled['auc']:.3f}  "
                  f"corr single={m_single['corr']:.3f} pooled={m_pooled['corr']:.3f}",
                  flush=True)

    np.savez(str(Path(args.out).with_suffix("")) + "_data.npz",
             rows=np.array([(r_[0], r_[1]) for r_ in rows], dtype=object),
             intervals=args.intervals, n_phases=args.n_phases,
             auc_single=[r_[2]["auc"] for r_ in rows],
             auc_pooled=[r_[3]["auc"] for r_ in rows],
             corr_single=[r_[2]["corr"] for r_ in rows],
             corr_pooled=[r_[3]["corr"] for r_ in rows],
             kinds=[r_[1] for r_ in rows], x=[r_[0] for r_ in rows])

    # ---- plot ---- #
    fs.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.13, wspace=0.25)
    colors = {"deconv": fs.ACCENT, "raw": "#e34948"}
    for kind in ("deconv", "raw"):
        xs = [r_[0] for r_ in rows if r_[1] == kind]
        auc_s = [r_[2]["auc"] for r_ in rows if r_[1] == kind]
        auc_p = [r_[3]["auc"] for r_ in rows if r_[1] == kind]
        corr_s = [r_[2]["corr"] for r_ in rows if r_[1] == kind]
        corr_p = [r_[3]["corr"] for r_ in rows if r_[1] == kind]
        c = colors[kind]
        axes[0].plot(xs, auc_s, "o--", color=c, alpha=0.55, label=f"{kind} (single phase)")
        axes[0].plot(xs, auc_p, "o-", color=c, lw=2.2, label=f"{kind} (phase-averaged)")
        axes[1].plot(xs, corr_s, "o--", color=c, alpha=0.55, label=f"{kind} (single phase)")
        axes[1].plot(xs, corr_p, "o-", color=c, lw=2.2, label=f"{kind} (phase-averaged)")
    for ax, ylab in zip(axes, ["ROC-AUC", "correlation"]):
        ax.set_xscale("log"); ax.set(xlabel="camera frame interval (ms)", ylabel=ylab,
                                     ylim=(0, 1.02))
        ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True)
        fs.despine(ax)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Does averaging over camera start-phase change the coarse-interval "
                 f"drop?  ({args.net}, N={N}, T={args.T_ms/1000:.0f}k ms, "
                 f"dashed=today's single-phase figure)",
                 fontsize=11.5, color=fs.INK, x=0.08, ha="left", y=0.98)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
