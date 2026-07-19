"""
Streaming data-length sweep: simulate ONE long recording per seed and infer
connectivity at several recording lengths, to test whether more data recovers
inference in the clean-AI regime (n1250ai) where correlations are weak.

Memory stays O(N^2): spikes are read sparsely, calcium/feed are built in
time-chunks, and the lag-pair moments (Cxx, Cyx) are accumulated incrementally
and snapshotted at each checkpoint. All five methods derive from those moments.

Outputs, per (T, seed), under results/<name>_T<Tms>k/seed<k>/:
    adj_true.npy, A_ols.npy, A_en.npy, A_lasso.npy, A_lassodale.npy, A_endale.npy

Usage:
  python scripts/wrapup_run_stream.py --net n1250ai --sweep 500000 1000000 2000000 --seeds 1 2 3
  python scripts/wrapup_run_stream.py --net n1250ai --sweep 20000 50000 --seeds 1   # local check
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calcium_ar.simulation.brunel_network import BrunelNetwork
from calcium_ar.experiments.streaming import stream_moments
from calcium_ar.solvers.from_moments import (
    ols_from_moments, fista_from_moments, dale_from_moments, strongest_entry_types)

from wrapup_run import build_cfg, NETS       # reuse the net presets

ALL_METHODS = ["ols", "ridge", "en", "lasso", "lassodale", "endale"]


def solve_selected(Cxx, Cyx, cfg, methods):
    """Solve only the requested methods (OLS/Ridge are one closed-form solve each;
    en/lasso/dale are the iterative FISTA ones — skip them at very large N)."""
    out = {}
    if "ols" in methods:
        out["ols"] = ols_from_moments(Cxx, Cyx)
    if "ridge" in methods:
        out["ridge"] = ols_from_moments(Cxx, Cyx, ridge=cfg["lam_l2"])
    A_en = None
    if "en" in methods or "endale" in methods:
        A_en = fista_from_moments(Cxx, Cyx, cfg["lam_l1"], cfg["lam_l2"], cfg["n_iter"])
        if "en" in methods:
            out["en"] = A_en
    if "lasso" in methods or "lassodale" in methods:
        A_lasso = fista_from_moments(Cxx, Cyx, cfg["lam_l1"], 0.0, cfg["n_iter"])
        if "lasso" in methods:
            out["lasso"] = A_lasso
        if "lassodale" in methods:
            out["lassodale"] = dale_from_moments(Cxx, Cyx, strongest_entry_types(A_lasso),
                                                 cfg["lam_l1"], 0.0, cfg["n_iter"] + 300)
    if "endale" in methods:
        out["endale"] = dale_from_moments(Cxx, Cyx, strongest_entry_types(A_en),
                                          cfg["lam_l1"], cfg["lam_l2"], cfg["n_iter"] + 300)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="n1250ai", choices=list(NETS))
    ap.add_argument("--sweep", type=float, nargs="+", required=True,
                    help="recording lengths in ms (e.g. 500000 1000000 2000000)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--chunk-ms", type=float, default=10000.0)
    ap.add_argument("--device", default=None,
                    help="None=CPU/numpy; 'cuda'=GPU accumulation (for large N)")
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS,
                    help="which estimators to solve (use 'ols ridge' at N=12500)")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--cache-dir", default=None,
                    help="cache ground-truth spikes here; reused instead of "
                         "re-running NEST when the same net/seed/length is found")
    ap.add_argument("--save-moments", action="store_true",
                    help="also save Cxx/Cyx per checkpoint (~2.5 GB at N=12500) so "
                         "new estimators can be solved in seconds without re-running")
    args = ap.parse_args()

    cfg = build_cfg(args.net)
    dt = cfg["dt"]
    lag = max(1, round(cfg["lag_ms"] / dt))
    w = max(5, round(cfg["smooth_window_ms"] / dt))
    smooth_win = w if w % 2 == 1 else w + 1
    max_T = max(args.sweep)
    checkpoints = [int(round(T / dt)) for T in args.sweep]
    chunk_samples = int(round(args.chunk_ms / dt))

    base = args.out_root or os.environ.get("CALCIUM_AR_WORKDIR") or str(ROOT)
    print(f"net={args.net} g={cfg['g']} eta={cfg['eta']}  sweep(ms)={args.sweep}  "
          f"seeds={args.seeds}  chunk_ms={args.chunk_ms}", flush=True)

    cache = Path(args.cache_dir) if args.cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        # --- ground truth: reuse the cached simulation when we already have it ---
        tag = f"{args.net}_seed{seed}_T{int(max_T)//1000}k"
        cf = (cache / f"{tag}.npz") if cache else None
        net = None
        if cf is not None and cf.exists():
            print(f"[seed {seed}] loading cached ground truth {cf.name} "
                  f"(skipping NEST)", flush=True)
            z = np.load(cf)
            spikes_cached = (z["idx"], z["times_ms"])
            adj_true = z["adj_true"].astype(np.float64)
        else:
            print(f"[seed {seed}] simulating {max_T:.0f} ms (densify=False) ...",
                  flush=True)
            net = BrunelNetwork(
                n_excitatory=cfg["n_excitatory"], n_inhibitory=cfg["n_inhibitory"],
                epsilon=cfg["epsilon"], g=cfg["g"], eta=cfg["eta"], J_ex=cfg["J_ex"],
                delay=cfg["delay"], V_reset=cfg["V_reset"],
                sim_time=max_T, dt=dt,
                n_threads=cfg["n_threads"], seed=seed)
            net.build(); net.run(densify=False)
            adj_true = net.get_adjacency()
            spikes_cached = net.get_spike_events()
            if cf is not None:
                idx_c, t_c = spikes_cached
                np.savez(cf, idx=idx_c.astype(np.int16), times_ms=t_c.astype(np.float32),
                         adj_true=adj_true.astype(np.float32))
                print(f"[seed {seed}] cached ground truth -> {cf}", flush=True)
        np.fill_diagonal(adj_true, 0.0)

        print(f"[seed {seed}] streaming moments + checkpoints ...", flush=True)

        cp_to_T = {int(round(T / dt)): T for T in args.sweep}

        def save_checkpoint(cp, Cxx, Cyx, _seed=seed, _adj=adj_true):
            """Solve + persist as soon as a checkpoint is reached, so a wall-clock
            timeout still leaves the earlier recording lengths on disk."""
            T = cp_to_T[cp]
            mats = solve_selected(Cxx, Cyx, cfg, args.methods)
            outdir = Path(base) / "results" / f"{cfg['name']}_T{int(T)//1000}k" / f"seed{_seed}"
            outdir.mkdir(parents=True, exist_ok=True)
            np.save(outdir / "adj_true.npy", _adj.astype(np.float32))
            for name, A in mats.items():
                np.save(outdir / f"A_{name}.npy", A.astype(np.float32))
            if args.save_moments:
                # every estimator derives from these two N x N matrices, so a
                # cached copy makes any NEW estimator a seconds-long job
                np.save(outdir / "Cxx.npy", Cxx)
                np.save(outdir / "Cyx.npy", Cyx)
            k0 = next(iter(mats))
            print(f"[seed {_seed}] T={T:.0f}ms SAVED -> {outdir}  "
                  f"|{k0}|max={np.abs(mats[k0]).max():.2e}", flush=True)

        moments, tau_est, rate = stream_moments(
            net, N=cfg["n_excitatory"] + cfg["n_inhibitory"], sim_time=max_T,
            spike_events=spikes_cached, dt=dt, lag=lag, tau=cfg["tau"], amplitude=cfg["amplitude"],
            sigma_intra=cfg["sigma_intra"], sigma_extra=cfg["sigma_extra"],
            smooth_win=smooth_win, tau_method=cfg["tau_method"],
            checkpoints_samples=checkpoints, chunk_samples=chunk_samples, seed=seed,
            device=args.device, on_checkpoint=save_checkpoint)
        print(f"[seed {seed}] mean rate {rate:.1f} Hz  (all checkpoints saved)", flush=True)

    print("\nsweep complete")


if __name__ == "__main__":
    main()
