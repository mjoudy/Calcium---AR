"""
Regime scan at N=1250: sweep (J_ex, eta) to find a clean asynchronous-irregular
(AI) state at a realistic firing rate, before scaling to full Brunel N=12500.

Theory (Brunel 2000): with g>4 (inhibition-dominated) the network is AI for
moderate external drive; the mean firing rate rises with eta. Shrinking Brunel
10x (N_E 10000->1000, so C_E = eps*N_E 1000->100) requires rescaling J to
preserve the *input fluctuations* that drive AI firing: fluctuation-preserving
scaling J*sqrt(C_E)=const gives J ~ 0.1*sqrt(10) ~ 0.32 (vs Brunel's 0.1). So we
sweep J across ~0.1-0.8 (bracketing 0.32 and the current 0.8) and eta ~1.2-4,
with g=5 fixed.

For each config (short sim, startup transient discarded) we measure:
  - firing rate E / I / all (Hz) and silent fraction
  - mean CV of ISI      (irregularity: AI ~ 1, regular ~ 0)
  - synchrony index     (mean pairwise spike-count correlation: AI ~ 0)
and an "AI quality" score that rewards CV~1, low synchrony, and a realistic rate.

Regime-only (no calcium, no inference), so it is cheap. The winners then go
through the full wrap-up pipeline (scripts/wrapup_run.py) separately.

Usage:
  python scripts/regime_scan.py                       # default grid, N=1250
  python scripts/regime_scan.py --sim-time 5000 --seeds 1
  python scripts/regime_scan.py --out $CALCIUM_AR_WORKDIR/results/regime_scan
"""

from __future__ import annotations

import argparse
import itertools
import os
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

# --- default grid (Brunel-theory motivated) ---------------------------------- #
# The AI state is fluctuation-driven: mean input near/below threshold, firing
# from fluctuations. With this LIF setup the external mean is ~eta*V_th, so eta
# must be near 1 (eta>1 => mean-driven, fast, regular). g sets how inhibition-
# dominated (=> irregular, low-rate) the state is; J sets fluctuation size and,
# too large, drives synchrony.
J_GRID = [0.2, 0.3, 0.5, 0.8]
ETA_GRID = [0.8, 0.9, 1.0, 1.1, 1.2]
G_GRID = [5.0, 6.0, 8.0]
N_E, N_I = 1000, 250
RATE_TARGET = 8.0          # Hz, "realistic cortical-ish" ideal (scored on log scale)
RATE_OK = (2.0, 20.0)      # acceptable band


def synchrony_index(spikes: np.ndarray, dt: float, bin_ms: float = 5.0,
                    n_sample: int = 150, seed: int = 0) -> float:
    """Mean pairwise spike-count correlation on a random active subset.
    ~0 = asynchronous (AI), positive = synchronous."""
    N, T = spikes.shape
    b = max(1, int(round(bin_ms / dt)))
    nb = T // b
    if nb < 2:
        return float("nan")
    counts = spikes[:, :nb * b].reshape(N, nb, b).sum(2)
    active = np.where(counts.sum(1) > 0)[0]
    if len(active) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.choice(active, size=min(n_sample, len(active)), replace=False)
    C = np.corrcoef(counts[idx])
    off = ~np.eye(len(idx), dtype=bool)
    v = C[off]
    return float(np.nanmean(v))


def ai_score(rate: float, cv: float, sync: float, silent_frac: float) -> float:
    """Heuristic 'clean AI at a realistic rate' score in [0,1].

    CV target is calibrated to what a finite (C_E=100) network can reach: CV>=0.8
    scores full, CV<=0.3 (regular) scores zero. Rewards realistic rate and low
    synchrony; product so a config must satisfy all three."""
    if not (RATE_OK[0] <= rate <= RATE_OK[1]) or silent_frac > 0.2:
        s_rate = 0.0
    else:
        s_rate = float(np.exp(-0.5 * (np.log(rate / RATE_TARGET) / 0.5) ** 2))
    s_cv = float(np.clip((cv - 0.3) / (0.8 - 0.3), 0.0, 1.0))   # 1 at CV>=0.8
    s_sync = float(np.clip(1.0 - abs(sync) / 0.10, 0.0, 1.0))    # 1 at sync=0, 0 at >=0.1
    return float(s_rate * s_cv * s_sync)


def run_one(J, eta, seed, sim_time, dt, warmup_ms, n_threads, g=5.0):
    net = BrunelNetwork(n_excitatory=N_E, n_inhibitory=N_I, epsilon=0.1,
                        g=g, eta=eta, J_ex=J, delay=1.5,
                        sim_time=sim_time, dt=dt, n_threads=n_threads, seed=seed)
    net.build(); net.run()
    spikes, _ = net.get_results()
    w = int(round(warmup_ms / dt))
    spk = spikes[:, w:]                                   # discard startup transient
    T_sec = spk.shape[1] * dt / 1000.0
    rates = spk.sum(1) / T_sec
    rE, rI = rates[:N_E], rates[N_E:]
    cv = calculate_cv(spk)
    cv_mean = float(np.nanmean(np.asarray(cv)[rates > 0])) if (rates > 0).any() else float("nan")
    sync = synchrony_index(spk, dt)
    silent = float((rates == 0).mean())
    return dict(J=J, eta=eta, g=g, seed=seed,
                rate_E=float(rE.mean()), rate_I=float(rI.mean()),
                rate_all=float(rates.mean()), silent_frac=silent,
                cv=cv_mean, sync=sync,
                score=ai_score(float(rates.mean()), cv_mean, sync, silent))


def make_figure(rows, out_png, g_label, global_best):
    Js = sorted({r["J"] for r in rows})
    Es = sorted({r["eta"] for r in rows})
    # average over seeds -> grid[eta_idx, J_idx]
    def grid(key):
        M = np.full((len(Es), len(Js)), np.nan)
        for i, e in enumerate(Es):
            for j, J in enumerate(Js):
                vals = [r[key] for r in rows if r["J"] == J and r["eta"] == e]
                if vals:
                    M[i, j] = np.nanmean(vals)
        return M

    panels = [("rate_all", "firing rate (Hz)", "viridis", None),
              ("cv", "CV of ISI  (AI~1)", "magma", None),
              ("sync", "synchrony  (AI~0)", "magma_r", None),
              ("score", "AI quality score", "Greens", (0, 1))]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=0.16, wspace=0.32)
    for ax, (key, title, cmap, lim) in zip(axes, panels):
        M = grid(key)
        vmin, vmax = (lim if lim else (np.nanmin(M), np.nanmax(M)))
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(Js))); ax.set_xticklabels([f"{J:g}" for J in Js])
        ax.set_yticks(range(len(Es))); ax.set_yticklabels([f"{e:g}" for e in Es])
        ax.set_xlabel("J_ex (mV)"); ax.set_ylabel("eta")
        ax.set_title(title, fontsize=10)
        for i, e in enumerate(Es):
            for j, J in enumerate(Js):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if key in ("sync",) else "black")
        # mark the GLOBAL best config if it lives on this g-slice
        if global_best["g"] == rows[0]["g"]:
            bi, bj = Es.index(global_best["eta"]), Js.index(global_best["J"])
            ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                                       edgecolor="#e34948", lw=2.4))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Regime scan N={N_E + N_I}, {g_label} — "
                 f"global best: g={global_best['g']:g}, J={global_best['J']:g}, "
                 f"eta={global_best['eta']:g}  (rate {global_best['rate_all']:.1f} Hz, "
                 f"CV {global_best['cv']:.2f}, sync {global_best['sync']:.3f})",
                 fontsize=12, x=0.05, ha="left")
    fig.savefig(out_png, dpi=140, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-time", type=float, default=5000.0, help="ms (short; stats only)")
    ap.add_argument("--warmup-ms", type=float, default=500.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--J", type=float, nargs="+", default=J_GRID)
    ap.add_argument("--eta", type=float, nargs="+", default=ETA_GRID)
    ap.add_argument("--g", type=float, nargs="+", default=G_GRID)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = os.environ.get("CALCIUM_AR_WORKDIR")
    out = Path(args.out) if args.out else (Path(base) if base else ROOT) / "results" / "regime_scan"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    combos = list(itertools.product(args.g, args.J, args.eta, args.seeds))
    for k, (g, J, eta, seed) in enumerate(combos, 1):
        r = run_one(J, eta, seed, args.sim_time, args.dt, args.warmup_ms, args.n_threads, g=g)
        rows.append(r)
        print(f"[{k}/{len(combos)}] g={g:g} J={J:g} eta={eta:g} seed={seed} | "
              f"rate={r['rate_all']:.1f}Hz (E{r['rate_E']:.1f}/I{r['rate_I']:.1f}) "
              f"CV={r['cv']:.2f} sync={r['sync']:.3f} silent={r['silent_frac']:.2f} "
              f"score={r['score']:.2f}", flush=True)

    # average score over seeds per (g,J,eta) to pick a robust global best
    import csv
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[(r["g"], r["J"], r["eta"])].append(r)
    best_key = max(agg, key=lambda k: np.nanmean([x["score"] for x in agg[k]]))
    best = {**agg[best_key][0]}
    for kk in ("rate_all", "cv", "sync", "score", "rate_E", "rate_I"):
        best[kk] = float(np.nanmean([x[kk] for x in agg[best_key]]))

    csv_path = out / "regime_scan.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    figs = []
    for g in args.g:
        g_rows = [r for r in rows if r["g"] == g]
        if g_rows:
            p = out / f"regime_scan_g{g:g}.png"
            make_figure(g_rows, p, f"g={g:g}", best)
            figs.append(p)

    print(f"\nwrote {csv_path}")
    for p in figs:
        print(f"wrote {p}")
    print(f"\nGLOBAL BEST: g={best['g']:g}, J={best['J']:g}, eta={best['eta']:g}"
          f"  -> rate {best['rate_all']:.1f} Hz (E{best['rate_E']:.1f}/I{best['rate_I']:.1f}), "
          f"CV {best['cv']:.2f}, sync {best['sync']:.3f}, score {best['score']:.2f}")

    # rank top 5 for the report
    ranked = sorted(agg.items(),
                    key=lambda kv: np.nanmean([x["score"] for x in kv[1]]), reverse=True)
    print("\nTop 5 configs (by AI-quality score):")
    for (g, J, eta), rs in ranked[:5]:
        sc = np.nanmean([x["score"] for x in rs]); rt = np.nanmean([x["rate_all"] for x in rs])
        cv = np.nanmean([x["cv"] for x in rs]); sy = np.nanmean([x["sync"] for x in rs])
        print(f"  g={g:g} J={J:g} eta={eta:g}: score {sc:.2f}  "
              f"(rate {rt:.1f} Hz, CV {cv:.2f}, sync {sy:.3f})")


if __name__ == "__main__":
    main()
