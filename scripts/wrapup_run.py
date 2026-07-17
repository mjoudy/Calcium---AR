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
    epsilon=0.1, delay=1.5, dt=0.1,
    tau=100.0, amplitude=1.0, sigma_intra=0.01, sigma_extra=0.05,
    smooth_window_ms=3.1, tau_method="ransac",
    lag_ms=2.0,                 # deconvolved-feed landscape peak (~synaptic delay)
    n_iter=500,
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
    # Full Brunel scale in the clean-AI regime. J fluctuation-scaled from n1250ai:
    # J*sqrt(C_E) const, C_E 100 -> 1250, so J = 0.8*sqrt(100/1250) ~ 0.23.
    "n12500ai": dict(n_excitatory=10000, n_inhibitory=2500, J_ex=0.23, g=8.0, eta=1.0,
                     sim_time=1_000_000.0, n_threads=16, lam_l1=1e-4, lam_l2=1e-4,
                     name="wrapup_n12500ai"),
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
        delay=cfg["delay"], sim_time=cfg["sim_time"], dt=dt,
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
