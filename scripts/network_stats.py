"""
Network diagnostics for one dataset: firing rates + raster + calcium traces.

Shows how a given network config is actually behaving. Reads a dataset dir
(spikes.zarr, calcium.zarr), prints E/I mean firing rates, and saves a 2-panel
figure (spike raster + calcium traces with spike marks).

Usage:
    python scripts/network_stats.py <dataset_dir> [--n-exc 1000] [--dt 0.1] [--out fig.png]
e.g. python scripts/network_stats.py $(ws_find calcium_ar)/data/N1250/s1_tau100_T50000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_panel import _raster, _calcium


def network_stats(data_path, n_exc=1000, dt=0.1, out=None):
    spikes = np.asarray(zarr.open(os.path.join(data_path, "spikes.zarr"), "r")[:])
    calcium = np.asarray(zarr.open(os.path.join(data_path, "calcium.zarr"), "r")[:])
    N, T = spikes.shape
    T_sec = T * dt / 1000.0
    rates = spikes.sum(1) / T_sec
    rE, rI = rates[:n_exc], rates[n_exc:]

    stats = {
        "N": N, "T_ms": T * dt,
        "rate_E": float(rE.mean()), "rate_I": float(rI.mean()),
        "rate_all": float(rates.mean()),
        "rate_std": float(rates.std()),
        "silent": int((rates == 0).sum()),
    }
    name = os.path.basename(os.path.normpath(data_path))
    print(f"[network_stats] {name}")
    print(f"  N={N}  T={T * dt:.0f} ms")
    print(f"  firing rate:  E={stats['rate_E']:.2f} Hz   I={stats['rate_I']:.2f} Hz   "
          f"all={stats['rate_all']:.2f} Hz (std {stats['rate_std']:.2f})")
    print(f"  silent neurons: {stats['silent']}/{N}")

    fig, ax = plt.subplots(2, 1, figsize=(12, 7))
    _raster(ax[0], spikes, dt)
    _calcium(ax[1], calcium, spikes, dt)
    fig.suptitle(f"{name}   |   E={stats['rate_E']:.1f} Hz, I={stats['rate_I']:.1f} Hz",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out or os.path.join(data_path, "network_stats.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("  wrote", out)
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="firing rates + raster/calcium for one dataset")
    p.add_argument("data_path")
    p.add_argument("--n-exc", type=int, default=1000)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    network_stats(a.data_path, a.n_exc, a.dt, a.out)
