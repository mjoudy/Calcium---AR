"""
R.2 follow-up: at a FIXED, realistic camera interval, does more recording
length (T) help -- and if so, does it plateau below the fast-camera ceiling
(the "fixes noise, not the blur" prediction) or fully recover?

One simulation at max(T), then each smaller T checkpoint reuses a PREFIX of
the same spike train (no need to resimulate per T) -- same trick as the R.4
streaming ladder, applied here without touching fig_r2_compute.py's slower
per-point pipeline (this script only calls its already-existing `run()`).

Usage:
  python scripts/fig_r2_longT.py --net n1250_r4 --interval-ms 33 \
      --Ts 100000 500000 1000000 2000000 5000000 --out figures/fig_R2_longT
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from calcium_ar.simulation.brunel_network import BrunelNetwork
from wrapup_run import build_cfg
from fig_r2_compute import run
import figstyle as fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="n1250_r4")
    ap.add_argument("--interval-ms", type=float, default=33.0)
    ap.add_argument("--Ts", type=float, nargs="+",
                    default=[100_000, 500_000, 1_000_000, 2_000_000, 5_000_000])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--out", default="figures/fig_R2_longT")
    args = ap.parse_args()

    cfg = build_cfg(args.net); dt = cfg["dt"]
    T_max = max(args.Ts)
    print(f"simulating {args.net} for {T_max:.0f} ms (one sim, reused as "
          f"prefixes for the shorter T checkpoints) ...", flush=True)
    net = BrunelNetwork(n_excitatory=cfg["n_excitatory"], n_inhibitory=cfg["n_inhibitory"],
                        epsilon=cfg["epsilon"], g=cfg["g"], eta=cfg["eta"], J_ex=cfg["J_ex"],
                        delay=cfg["delay"], V_reset=cfg["V_reset"], sim_time=T_max,
                        dt=dt, n_threads=cfg["n_threads"], seed=args.seed)
    net.build(); net.run(densify=False)
    idx_full, tms_full = net.get_spike_events(); idx_full = idx_full.astype(np.int64)
    adj = net.get_adjacency(); np.fill_diagonal(adj, 0.0)
    N = adj.shape[0]; n_exc = int(cfg["n_excitatory"])

    rows = []
    for T in sorted(args.Ts):
        sel = tms_full < T
        idx, tms = idx_full[sel], tms_full[sel]
        for kind in ("deconv_rate", "raw_rate"):
            t0 = time.time()
            _, _, mets, _ = run(kind, args.interval_ms, idx, tms, N, dt, adj, T,
                                args.chunk, args.seed, n_exc, density=args.density)
            dt_wall = time.time() - t0
            rows.append((T, kind, mets["auc"], mets["corr"]))
            print(f"T={T:<10.0f} {kind:12s} AUC={mets['auc']:.3f} "
                  f"corr={mets['corr']:.3f}  ({dt_wall:.0f}s)", flush=True)
        np.savez(str(Path(args.out).with_suffix("")) + "_data.npz",
                 rows=np.array([(r[0], r[1], r[2], r[3]) for r in rows], dtype=object),
                 interval_ms=args.interval_ms)

    fs.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.85, bottom=0.15, wspace=0.25)
    colors = {"deconv_rate": fs.ACCENT, "raw_rate": "#e34948"}
    labels = {"deconv_rate": "deconvolved", "raw_rate": "raw calcium"}
    for kind in ("deconv_rate", "raw_rate"):
        xs = [r[0] for r in rows if r[1] == kind]
        aucs = [r[2] for r in rows if r[1] == kind]
        corrs = [r[3] for r in rows if r[1] == kind]
        axes[0].plot(xs, aucs, "o-", color=colors[kind], lw=2, label=labels[kind])
        axes[1].plot(xs, corrs, "o-", color=colors[kind], lw=2, label=labels[kind])
    for ax, ylab in zip(axes, ["ROC-AUC", "correlation"]):
        ax.set_xscale("log"); ax.set(xlabel="recording length T (ms)", ylabel=ylab,
                                     ylim=(0, 1.02))
        ax.axhline(0.5 if ylab == "ROC-AUC" else 0, color=fs.GRID, lw=1, ls=":")
        ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True)
        fs.despine(ax)
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle(f"Does more data help at a fixed, realistic camera interval "
                 f"({args.interval_ms:g} ms)?  ({args.net}, N={N})",
                 fontsize=12.5, color=fs.INK, x=0.09, ha="left", y=0.97)
    fs.save(fig, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
