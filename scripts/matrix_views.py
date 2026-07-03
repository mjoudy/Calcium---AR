"""
Four views of the connectivity matrix (ground truth vs estimated) for one run.

Columns:
  1. signed weights   — as in the panel; magnitude-dominated, so |I|>>|E| makes the
                        inhibitory band look solid even though both are ~10% dense.
  2. sign only        — +1 / 0 / -1 (E=red, none=white, I=blue): shows structure + sign
                        with NO magnitude, so the true density is equal for both types.
  3. column-normalized — each column divided by its std, so weak E and strong I columns
                        render at comparable brightness.
  4. sub-block         — a zoom spanning the E/I source boundary; individual connections
                        are resolvable here even at large N.

Rows: ground truth, estimated.

Usage: python scripts/matrix_views.py <run_dir> [--stage EN|EN+Dale|EN+Dale+balance] [--out fig.png]
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _signed(ax, A, title):
    nz = np.abs(A[A != 0])
    v = np.percentile(nz, 99) if nz.size else 1.0
    im = ax.imshow(A, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set(title=title, xlabel="source", ylabel="target")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def _sign(ax, A, title, eps=1e-9):
    S = np.sign(A).astype(float)
    S[np.abs(A) <= eps] = 0.0
    ax.imshow(S, cmap="bwr", vmin=-1, vmax=1, aspect="auto")
    ax.set(title=title, xlabel="source", ylabel="target")


def _colnorm(ax, A, title):
    B = A.astype(float).copy()
    sd = B.std(0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    B = B / sd
    nz = np.abs(B[B != 0])
    v = np.percentile(nz, 99) if nz.size else 1.0
    ax.imshow(B, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set(title=title, xlabel="source", ylabel="target")


def _subblock(ax, A, title, n_exc, half=50):
    N = A.shape[0]
    c0, c1 = max(0, n_exc - half), min(N, n_exc + half)
    r1 = min(N, 2 * half)
    sub = A[0:r1, c0:c1]
    nz = np.abs(sub[sub != 0])
    v = np.percentile(nz, 99) if nz.size else 1.0
    ax.imshow(sub, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto", extent=[c0, c1, r1, 0])
    ax.axvline(n_exc, color="k", lw=0.6)
    ax.set(title=title, xlabel="source  (E | I)", ylabel="target")


def render(adj, est, title, out, n_exc=1000):
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    for r, (A, nm) in enumerate([(adj, "GT"), (est, "estimated")]):
        _signed(ax[r, 0], A, f"{nm}: signed weights")
        _sign(ax[r, 1], A, f"{nm}: sign only (E=red, I=blue)")
        _colnorm(ax[r, 2], A, f"{nm}: column-normalized")
        _subblock(ax[r, 3], A, f"{nm}: sub-block (E | I)", n_exc)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def matrix_views(run_dir, stage="EN+Dale", out=None):
    from calcium_ar.experiments.config import ExperimentConfig

    run_dir = Path(run_dir)
    config = ExperimentConfig.from_json(run_dir / "config.json")
    adj = np.load(run_dir / "adj_true.npy").T          # align to estimate convention

    mats = {"EN": "adj_inferred.npy", "EN+Dale": "adj_dale.npy",
            "EN+Dale+balance": "adj_dale_balance.npy"}
    p = run_dir / mats.get(stage, "adj_inferred.npy")
    if not p.exists():
        p, stage = run_dir / "adj_inferred.npy", "EN"
    est = np.load(p)

    out = out or str(run_dir / f"matrix_views_{stage.replace('+', '_')}.png")
    title = f"{config.name}  [{stage}]   N={adj.shape[0]}"
    return render(adj, est, title, out, config.n_excitatory)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="four matrix views for one run")
    p.add_argument("run_dir")
    p.add_argument("--stage", default="EN+Dale",
                   choices=["EN", "EN+Dale", "EN+Dale+balance"])
    p.add_argument("--out", default=None)
    a = p.parse_args()
    print("wrote", matrix_views(a.run_dir, a.stage, a.out))
