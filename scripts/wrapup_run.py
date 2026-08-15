"""
Wrap-up: local run (small net) that produces the estimates the figures need.

Reuses the existing N=100 configuration (NE=80, NI=20, J_ex=10) at the validated
landscape settings (deconvolved feed, lag=2 ms). For each seed it builds one
network, makes one shared feed, and estimates connectivity with all five methods
so they are compared on identical data:

    OLS, EN (Elastic Net), Lasso (pure L1), Lasso+Dale, EN+Dale

Balance (E/I magnitude rescale) is intentionally left out of this wrap-up.

Outputs, per seed, under results/wrapup_local/seed{k}/:
    adj_true.npy, A_ols.npy, A_en.npy, A_lasso.npy, A_lassodale.npy, A_endale.npy

Usage:  python scripts/wrapup_run.py                       # N=100 local, seeds 1 2 3
        python scripts/wrapup_run.py --net n1250 --seeds 1 2 3 4 5
        python scripts/wrapup_run.py --net n1250 --out /path/to/results/wrapup_n1250

Big outputs go to $CALCIUM_AR_WORKDIR/results/<name> on the cluster (env set by the
SLURM job), falling back to ./results/<name> when run locally.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calcium_ar.simulation.brunel_network import BrunelNetwork
from calcium_ar.simulation.calcium_signal import simulate_calcium
from calcium_ar.preprocessing.signal_utils import get_signal_derivative_pair
from calcium_ar.preprocessing.tau_estimation import estimate_tau_robust
from calcium_ar.preprocessing.feed_reconstruction import reconstruct_feed
from calcium_ar.solvers import solve_ols, solve_fista
from calcium_ar.solvers.dale_fista import dale_fista

# --- Settings shared across nets (validated landscape) ----------------------- #
COMMON = dict(
    epsilon=0.1, delay=1.5, dt=0.1, V_reset=0.0,
    tau=100.0, amplitude=1.0, sigma_intra=0.01, sigma_extra=0.05,
    smooth_window_ms=3.1, tau_method="ransac",
    lag_ms=2.0,                 # deconvolved-feed landscape peak (~synaptic delay)
    n_iter=500,
    tau_m=20.0,                 # membrane time const (ms); per-net override for PIF-like nets
)

# Per-net overrides. n100 reuses the existing small-net setup (few-synapse J_ex=10);
# n1250 is the old baseline (g=5, eta=2 -> high rate ~42 Hz, regular); n1250ai is
# the regime-scan winner: g=8, eta=1.0 -> clean asynchronous-irregular at a
# realistic ~8 Hz (CV 0.80, synchrony 0.027). See scripts/regime_scan.py.
#
# g/eta are per-net (they define the dynamical regime). lam is per-net too: L1
# strength must scale with weight magnitude -- at J_ex=10 (n100) lam_l1=3e-3 is
# fine; at J_ex=0.8 it must drop to ~1e-4 or every regularized method collapses
# to zero.
NETS = {
    "n100":    dict(n_excitatory=80,   n_inhibitory=20,  J_ex=10.0, g=5.0, eta=2.0,
                    sim_time=5000.0,  n_threads=4, lam_l1=3e-3, lam_l2=1e-3,
                    name="wrapup_local"),
    "n1250":   dict(n_excitatory=1000, n_inhibitory=250, J_ex=0.8,  g=5.0, eta=2.0,
                    sim_time=50000.0, n_threads=8, lam_l1=1e-4, lam_l2=1e-4,
                    name="wrapup_n1250"),
    "n1250ai": dict(n_excitatory=1000, n_inhibitory=250, J_ex=0.8,  g=8.0, eta=1.0,
                    sim_time=50000.0, n_threads=8, lam_l1=1e-4, lam_l2=1e-4,
                    name="wrapup_n1250ai"),
    # Full Brunel scale, our tuned variant (J fluctuation-scaled from n1250ai,
    # V_reset=0). Kept for reproducibility of the runs already done.
    "n12500ai": dict(n_excitatory=10000, n_inhibitory=2500, J_ex=0.23, g=8.0, eta=1.0,
                     V_reset=0.0, sim_time=1_000_000.0, n_threads=16,
                     lam_l1=1e-4, lam_l2=1e-4, name="wrapup_n12500ai"),
    # CANONICAL Brunel 2000 at its native size — no rescaling needed here, which
    # is why "canonical" is only strictly meaningful at N=12500:
    #   N_E=10000, N_I=2500, eps=0.1 (C_E=1000), J=0.1 mV,
    #   delay=1.5, tau_m=20, theta=20, V_r=10, t_ref=2.
    # (g, eta) = (6, 4) is Brunel Fig 8B — the ASYNCHRONOUS IRREGULAR state.
    # NOTE: (5, 2) is Fig 8C = synchronous irregular, which is what we measured
    # as CV 0.43 — regular-ish, not AI. Use 8B for AI.
    "n12500": dict(n_excitatory=10000, n_inhibitory=2500, J_ex=0.1, g=6.0, eta=4.0,
                   V_reset=10.0, sim_time=1_000_000.0, n_threads=16,
                   lam_l1=1e-4, lam_l2=1e-4, name="wrapup_n12500"),
    # Same canonical parameters, but eta lowered to put the AI state at a
    # cortical-like rate. Balance gives nu ~ 20*(eta-1), so eta=1.5 -> ~10 Hz
    # (Fig 8B's eta=4 gives ~60 Hz). g/J/V_r/delay all stay canonical.
    "n12500_lowrate": dict(n_excitatory=10000, n_inhibitory=2500, J_ex=0.1,
                           g=6.0, eta=1.5, V_reset=10.0, sim_time=1_000_000.0,
                           n_threads=16, lam_l1=1e-4, lam_l2=1e-4,
                           name="wrapup_n12500lr"),
    # --- R.4 scaling ladder -------------------------------------------------- #
    # Same regime at every size so N is the ONLY variable: g=8, V_reset=10,
    # J fluctuation-scaled (J*sqrt(C_E) = 1.5*0.1*sqrt(1000) = 4.743), and eta
    # tuned per size to ~14 Hz (scripts/r4_tune_regime.py --g 8 --j-scale 1.5).
    #
    # VERIFIED (4 s sims, 1 s warm-up, seed 1):
    #   N= 1250  J=0.474  eta=1.30 -> 14.8 Hz  CV 0.98  sync 0.010
    #   N= 2500  J=0.335  eta=1.50 -> 13.9 Hz  CV 0.98  sync 0.008
    #   N= 5000  J=0.237  eta=1.90 -> 14.4 Hz  CV 1.03  sync 0.007
    #   N=12500  J=0.150  eta=2.60 -> 13.9 Hz  CV 1.06  sync 0.007
    #
    # Why g=8 and not Brunel's g=6: at g=6 with J*sqrt(C_E)=3.162 the same ~14 Hz
    # comes out at CV ~0.63 — asynchronous but only moderately irregular. The 2-D
    # probe (results/regime2d/) showed CV rises with BOTH g and J at fixed rate,
    # while synchrony FALLS with g, so this family is AI on both counts. It is a
    # deliberate departure from the canonical point, which lives in n12500 above.
    "n1250_r4": dict(n_excitatory=1000, n_inhibitory=250, J_ex=0.474, g=8.0,
                     eta=1.30, V_reset=10.0, sim_time=500_000.0, n_threads=8,
                     lam_l1=1e-4, lam_l2=1e-4, name="wrapup_n1250r4"),
    # --- PIF-like pilot: SAME topology/scale as n1250_r4, but tau_m x10 (20ms
    # -> 200ms) to approximate a "high input resistance" perfect-integrator
    # neuron (removes most of the leak nonlinearity; the spike threshold/reset
    # is still there, so this is "more linear", not fully linear -- see the
    # exact-linear OU test for that). eta re-tuned by direct probe (2026-08-14,
    # scratch script, not committed) since nu_th scales with tau_m: eta=8.0 ->
    # 13.7 Hz, sync 0.004 (AI-like), but CV=1.94 (much more irregular than
    # n1250_r4's CV~0.98 at the same rate) -- note this, don't assume it away.
    # SMALL-SCALE PILOT: sim_time kept short on purpose, extend only if the
    # shared-input signal looks worth a full production run.
    "n1250_pif": dict(n_excitatory=1000, n_inhibitory=250, J_ex=0.474, g=8.0,
                      eta=8.0, V_reset=10.0, tau_m=200.0, sim_time=100_000.0,
                      n_threads=8, lam_l1=1e-4, lam_l2=1e-4, name="wrapup_n1250pif"),
    # --- PIF ladder continued: tau_m x100 (20ms -> 2000ms). eta re-tuned by
    # scripts/pif_tau_probe.py --stage x100 (2026-08-15, sim_time=40000ms,
    # properly stationary per the 20x-tau_m floor -- an earlier provisional
    # probe at sim_time=4000ms is NOT what this eta comes from): eta=80.0 ->
    # 14.8 Hz, sync 0.004, but CV=6.14 -- MUCH more irregular again than the
    # x10 pilot's CV=1.94 (0.98 -> 1.94 -> 6.14 across x1/x10/x100). Expected
    # direction (perfect integrators fire more irregularly under fluctuation-
    # driven input, no leak to regularize ISIs) but the SIZE of the jump is
    # worth flagging every time this arm is discussed -- single-neuron
    # statistics are now far from realistic cortical AI (CV~1).
    # SMALL-SCALE PILOT: sim_time matches n1250_pif's (100s = 50x tau_m, well
    # past the stationarity floor); extend only if the signal looks worth it.
    "n1250_pif100": dict(n_excitatory=1000, n_inhibitory=250, J_ex=0.474, g=8.0,
                         eta=80.0, V_reset=10.0, tau_m=2000.0, sim_time=100_000.0,
                         n_threads=8, lam_l1=1e-4, lam_l2=1e-4, name="wrapup_n1250pif100"),
    "n2500_r4": dict(n_excitatory=2000, n_inhibitory=500, J_ex=0.335, g=8.0,
                     eta=1.50, V_reset=10.0, sim_time=1_000_000.0, n_threads=8,
                     lam_l1=1e-4, lam_l2=1e-4, name="wrapup_n2500r4"),
    "n5000_r4": dict(n_excitatory=4000, n_inhibitory=1000, J_ex=0.237, g=8.0,
                     eta=1.90, V_reset=10.0, sim_time=2_000_000.0, n_threads=8,
                     lam_l1=1e-4, lam_l2=1e-4, name="wrapup_n5000r4"),
    # top rung: the old n12500_lowrate run (g=6, J=0.1) is NOT part of this
    # ladder any more and has to be re-simulated at the new family.
    "n12500_r4": dict(n_excitatory=10000, n_inhibitory=2500, J_ex=0.150, g=8.0,
                      eta=2.60, V_reset=10.0, sim_time=5_000_000.0, n_threads=16,
                      lam_l1=1e-4, lam_l2=1e-4, name="wrapup_n12500r4"),

    # --- R.4b: FIXED IN-DEGREE ladder (alternative to fixed-probability) ----- #
    # The n*_r4 ladder above holds connection PROBABILITY fixed (epsilon=0.1),
    # so in-degree C_E=eps*N_E GROWS with N (100 -> 1000 across the ladder) --
    # more shared-input pathways per neuron at bigger N, which is the likely
    # driver of the excitatory-recall gap found in R.4. Real cortex is closer to
    # the OPPOSITE convention: a neuron's number of synaptic inputs is capped by
    # its dendritic tree, roughly independent of total local population size --
    # i.e. FIXED in-degree, with probability shrinking as N grows. This ladder
    # tests that: C_E is held at 100 (n1250_r4's own value) at every size, so
    # epsilon = 100/N_E shrinks (0.1 -> 0.01). Because C_E is now constant, J
    # does NOT need to be rescaled by size either (same J=0.474 everywhere,
    # unlike the r4 ladder) -- under the fluctuation-scaling convention, fixed
    # C_E + fixed J should give near-N-INVARIANT local dynamics, so eta=1.30
    # (n1250_r4's tuned value) is the theoretical prediction for every size too.
    # n1250_ci is IDENTICAL to n1250_r4 (epsilon=0.1 either way) -- it's the
    # shared anchor point, not a new simulation.
    # VERIFY WITH A PREFLIGHT before trusting the full run (see the SLURM jobs) --
    # this eta prediction is theory, not yet measured for N>1250.
    "n1250_ci": dict(n_excitatory=1000, n_inhibitory=250, epsilon=0.1,
                     J_ex=0.474, g=8.0, eta=1.30, V_reset=10.0,
                     sim_time=500_000.0, n_threads=8,
                     lam_l1=1e-4, lam_l2=1e-4, name="wrapup_n1250ci"),
    "n2500_ci": dict(n_excitatory=2000, n_inhibitory=500, epsilon=0.05,
                     J_ex=0.474, g=8.0, eta=1.30, V_reset=10.0,
                     sim_time=1_000_000.0, n_threads=8,
                     lam_l1=1e-4, lam_l2=1e-4, name="wrapup_n2500ci"),
    "n5000_ci": dict(n_excitatory=4000, n_inhibitory=1000, epsilon=0.025,
                     J_ex=0.474, g=8.0, eta=1.30, V_reset=10.0,
                     sim_time=2_000_000.0, n_threads=8,
                     lam_l1=1e-4, lam_l2=1e-4, name="wrapup_n5000ci"),
    "n12500_ci": dict(n_excitatory=10000, n_inhibitory=2500, epsilon=0.01,
                      J_ex=0.474, g=8.0, eta=1.30, V_reset=10.0,
                      sim_time=5_000_000.0, n_threads=16,
                      lam_l1=1e-4, lam_l2=1e-4, name="wrapup_n12500ci"),
}


def build_cfg(net: str) -> dict:
    if net not in NETS:
        raise SystemExit(f"unknown net '{net}'; choose from {list(NETS)}")
    return {**COMMON, **NETS[net]}


def strongest_entry_types(A: np.ndarray) -> np.ndarray:
    """Per-neuron sign from its strongest outgoing (column) weight; +1 default."""
    N = A.shape[0]
    t = np.ones(N)
    for j in range(N):
        c = A[:, j].copy(); c[j] = 0.0
        if np.any(c):
            t[j] = np.sign(c[np.argmax(np.abs(c))]) or 1.0
    return t


def make_feed(seed: int, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build one network + shared deconvolved feed. Returns (feed, adj_true)."""
    dt = cfg["dt"]
    net = BrunelNetwork(
        n_excitatory=cfg["n_excitatory"], n_inhibitory=cfg["n_inhibitory"],
        epsilon=cfg["epsilon"], g=cfg["g"], eta=cfg["eta"], J_ex=cfg["J_ex"],
        delay=cfg["delay"], V_reset=cfg["V_reset"], tau_m=cfg.get("tau_m", 20.0),
        sim_time=cfg["sim_time"], dt=dt,
        n_threads=cfg["n_threads"], seed=seed,
    )
    net.build(); net.run()
    spikes, adj_true = net.get_results()
    np.fill_diagonal(adj_true, 0.0)

    calcium = simulate_calcium(
        spikes, tau=cfg["tau"], dt=dt, amplitude=cfg["amplitude"],
        sigma_intra=cfg["sigma_intra"], sigma_extra=cfg["sigma_extra"], seed=seed,
    )

    w = max(5, round(cfg["smooth_window_ms"] / dt))
    smooth_win = w if w % 2 == 1 else w + 1
    smooth, deriv = get_signal_derivative_pair(calcium, window_length=smooth_win, delta=dt)
    tau_est = estimate_tau_robust(calcium, window_length=smooth_win,
                                  method=cfg["tau_method"], dt=dt)
    feed = reconstruct_feed(smooth, deriv, tau_est)
    return np.asarray(feed, dtype=float), adj_true


def estimate_all(feed: np.ndarray, seed_dir: Path, cfg: dict) -> dict[str, np.ndarray]:
    """Run all five methods on the shared feed. Returns {name: A}."""
    dt = cfg["dt"]
    lag = max(1, round(cfg["lag_ms"] / dt))

    # solvers read a zarr; write the shared feed once
    fz = seed_dir / "feed.zarr"
    if fz.exists():
        shutil.rmtree(fz)
    z = zarr.open(str(fz), mode="w", shape=feed.shape, chunks=(feed.shape[0], 10000),
                  dtype="f8")
    z[:] = feed

    A_ols = solve_ols(str(fz), lag=lag); np.fill_diagonal(A_ols, 0.0)
    A_en = solve_fista(str(fz), lag=lag, lam_l1=cfg["lam_l1"], lam_l2=cfg["lam_l2"],
                       n_iter=cfg["n_iter"]); np.fill_diagonal(A_en, 0.0)
    A_lasso = solve_fista(str(fz), lag=lag, lam_l1=cfg["lam_l1"], lam_l2=0.0,
                          n_iter=cfg["n_iter"]); np.fill_diagonal(A_lasso, 0.0)

    # Dale needs centred lag pairs in memory
    Xc = feed[:, :-lag] - feed[:, :-lag].mean(1, keepdims=True)
    Yc = feed[:, lag:] - feed[:, lag:].mean(1, keepdims=True)
    A_endale = dale_fista(Xc, Yc, strongest_entry_types(A_en),
                          lam1=cfg["lam_l1"], lam2=cfg["lam_l2"], n_iter=cfg["n_iter"] + 300)
    A_lassodale = dale_fista(Xc, Yc, strongest_entry_types(A_lasso),
                             lam1=cfg["lam_l1"], lam2=0.0, n_iter=cfg["n_iter"] + 300)

    shutil.rmtree(fz)
    return dict(ols=A_ols, en=A_en, lasso=A_lasso,
                lassodale=A_lassodale, endale=A_endale)


def resolve_out(cfg: dict, override: str | None) -> Path:
    if override:
        return Path(override)
    base = os.environ.get("CALCIUM_AR_WORKDIR")
    root = Path(base) if base else ROOT
    return root / "results" / cfg["name"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="n100", choices=list(NETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default=None, help="output dir (default: workspace or ./results)")
    args = ap.parse_args()

    cfg = build_cfg(args.net)
    out = resolve_out(cfg, args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"net={args.net}  N={cfg['n_excitatory'] + cfg['n_inhibitory']}  "
          f"seeds={args.seeds}  out={out}", flush=True)

    for seed in args.seeds:
        seed_dir = out / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        print(f"[seed {seed}] building network + feed ...", flush=True)
        feed, adj_true = make_feed(seed, cfg)
        np.save(seed_dir / "adj_true.npy", adj_true)
        print(f"[seed {seed}] estimating (OLS, EN, Lasso, Lasso+Dale, EN+Dale) ...",
              flush=True)
        mats = estimate_all(feed, seed_dir, cfg)
        for name, A in mats.items():
            np.save(seed_dir / f"A_{name}.npy", A)
        del feed
        dens = float((adj_true != 0).sum()) / (adj_true.size - adj_true.shape[0])
        print(f"[seed {seed}] done. true density={dens:.3f}  "
              f"nz(lasso)={int((mats['lasso'] != 0).sum())}", flush=True)

    print(f"\nAll seeds written under {out}")


if __name__ == "__main__":
    main()
