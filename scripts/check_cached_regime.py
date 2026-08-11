"""
Measure rate / ISI-CV / synchrony from the ACTUAL cached ground-truth spikes
used in a production run (not a separate short preflight sim) -- so the
reported dynamics are guaranteed to be the exact network instance behind the
results, not a same-parameters-different-seed proxy.

Reuses wrapup_run_stream.py's cache-lookup logic (same sim_spec -> same hash)
to find the .npz, then r4_tune_regime.py's sparse_stats() on a warmed-up
window of the cached spikes. No new simulation.

Usage:
  python scripts/check_cached_regime.py --net n2500_ci --seed 1 --max-t 5000000 \
      --cache-dir $(ws_find calcium_ar)/gt_cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from wrapup_run import build_cfg
from r4_tune_regime import sparse_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-t", type=float, required=True,
                    help="the sweep's max recording length (ms) -- this is what the cache key is keyed on")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--warmup-ms", type=float, default=1000.0)
    ap.add_argument("--window-ms", type=float, default=10000.0,
                    help="how much of the (post-warmup) cached recording to measure stats on")
    args = ap.parse_args()

    cfg = build_cfg(args.net)
    dt = cfg["dt"]
    sim_spec = dict(
        n_excitatory=cfg["n_excitatory"], n_inhibitory=cfg["n_inhibitory"],
        epsilon=cfg["epsilon"], g=cfg["g"], eta=cfg["eta"], J_ex=cfg["J_ex"],
        delay=cfg["delay"], V_reset=cfg["V_reset"], dt=dt,
        sim_time=args.max_t, seed=args.seed, n_threads=cfg["n_threads"],
    )
    spec_str = json.dumps(sim_spec, sort_keys=True)
    spec_hash = hashlib.sha1(spec_str.encode()).hexdigest()[:10]
    tag = f"{args.net}_seed{args.seed}_T{int(args.max_t)//1000}k_{spec_hash}"
    cf = Path(args.cache_dir) / f"{tag}.npz"
    if not cf.exists():
        raise SystemExit(f"cache file not found: {cf}\n"
                          f"(check --net/--seed/--max-t match a run that actually happened)")

    z = np.load(cf, allow_pickle=False)
    idx, times = z["idx"].astype(np.int64), z["times_ms"].astype(np.float64)
    N = cfg["n_excitatory"] + cfg["n_inhibitory"]
    print(f"loaded {cf.name}  ({len(times):,} spikes total, N={N}, eps={cfg['epsilon']})")

    T_window = args.warmup_ms + args.window_ms
    keep = times < T_window
    r = sparse_stats(idx[keep], times[keep], N, T_window, args.warmup_ms)
    print(f"\n{args.net} seed={args.seed}  (measured on {args.window_ms/1000:.0f}s after "
          f"{args.warmup_ms/1000:.0f}s warmup, from the ACTUAL cached production spikes)")
    print(f"  rate={r['rate']:.1f} Hz   CV={r['cv']:.3f}   sync={r['sync']:.4f}   "
          f"silent_frac={r['silent']:.3f}")


if __name__ == "__main__":
    main()
