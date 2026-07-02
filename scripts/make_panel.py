"""
7-panel diagnostic figure for one estimation run.

Panels:
  1. spike raster              — is the network firing (asynchronous-irregular)?
  2. calcium traces + spikes   — the signal we infer from
  3. ground-truth matrix
  4. estimated matrix
  5. three-class weight distribution (estimated weights split by true E / none / I)
  6. eigenvalue spectrum (estimated vs true)
  7. inferred-vs-true weights for a few random neurons

Reads the matrices from a run dir and the spikes/calcium from its dataset
(resolved via the run's config.json).

Usage:
    python scripts/make_panel.py <run_dir> [--stage EN|EN+Dale|EN+Dale+balance] [--out fig.png]
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


# --------------------------------------------------------------------------- #
# individual panels (all take a matplotlib axis + arrays -> easy to test)
# --------------------------------------------------------------------------- #

def _raster(ax, spikes, dt, n_show=120, t_ms=1000.0):
    N, T = spikes.shape
    t_end = min(T, max(1, int(t_ms / dt)))
    idx = np.unique(np.linspace(0, N - 1, min(n_show, N)).astype(int))
    for row, j in enumerate(idx):
        ts = np.where(spikes[j, :t_end] > 0)[0] * dt
        if ts.size:
            ax.plot(ts, np.full(ts.shape, row, float), "|", color="k", ms=3, mew=0.5)
    ax.set(title="spike raster", xlabel="time (ms)", ylabel="neuron (subset)")


def _calcium(ax, calcium, spikes, dt, n_show=3, t_ms=2000.0):
    N, T = calcium.shape
    t_end = min(T, max(1, int(t_ms / dt)))
    t = np.arange(t_end) * dt
    for k in range(min(n_show, N)):
        y = calcium[k, :t_end]
        off = k * (np.ptp(y) + 1e-9)
        ax.plot(t, y + off, lw=0.6)
        sp = np.where(spikes[k, :t_end] > 0)[0] * dt
        if sp.size:
            ax.plot(sp, np.full(sp.shape, y.min() + off), ".", ms=2, color="r")
    ax.set(title="calcium traces + spikes", xlabel="time (ms)", ylabel="ΔF (offset)")


def _matrix(ax, A, title):
    nz = np.abs(A[A != 0])
    v = np.percentile(nz, 99) if nz.size else 1.0
    im = ax.imshow(A, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set(title=title, xlabel="source", ylabel="target")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def _three_class(ax, est, adj_true):
    off = ~np.eye(adj_true.shape[0], dtype=bool)
    g = adj_true[off]
    e = est[off]
    for mask, lab, c in [(g > 0, "true E", "tab:red"),
                         (g == 0, "true none", "0.5"),
                         (g < 0, "true I", "tab:blue")]:
        vals = e[mask]
        if vals.size:
            ax.hist(vals, bins=60, alpha=0.55, label=lab, color=c, density=True)
    ax.set(title="estimated weight by true class", xlabel="estimated weight", ylabel="density")
    ax.legend(fontsize=7)


def _eigs(ax, est, adj_true, max_n=5000):
    N = adj_true.shape[0]
    if N > max_n:
        ax.text(0.5, 0.5, f"eigvals skipped\n(N={N} > {max_n})",
                ha="center", va="center", transform=ax.transAxes)
        ax.set(title="eigenvalue spectrum")
        return
    for A, c, lab in [(adj_true, "0.6", "true"), (est, "tab:green", "estimated")]:
        w = np.linalg.eigvals(A)
        ax.scatter(w.real, w.imag, s=6, alpha=0.5, color=c, label=lab)
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set(title="eigenvalue spectrum", xlabel="Re", ylabel="Im")
    ax.legend(fontsize=7)


def _confusion(ax, est, adj_true, eps=1e-9):
    """3-class confusion matrix (true vs predicted: E / unconnected / I) over
    off-diagonal entries. Prediction = sign of the estimated weight (0 = none)."""
    off = ~np.eye(adj_true.shape[0], dtype=bool)

    def lab(A):
        s = np.sign(A[off]).astype(int)
        s[np.abs(A[off]) <= eps] = 0
        return s

    t, p = lab(adj_true), lab(est)
    classes, names = [1, 0, -1], ["E", "none", "I"]
    M = np.array([[int(np.sum((t == ct) & (p == cp))) for cp in classes]
                  for ct in classes])
    ax.imshow(M, cmap="Blues")
    ax.set(xticks=[0, 1, 2], yticks=[0, 1, 2], xticklabels=names, yticklabels=names,
           xlabel="predicted", ylabel="true", title="confusion (3-class)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:,}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > M.max() / 2 else "black")


def _random_neurons(ax, est, adj_true, n=4, seed=0):
    rng = np.random.default_rng(seed)
    N = adj_true.shape[0]
    js = rng.choice(N, size=min(n, N), replace=False)
    lo = hi = 0.0
    for j in js:
        xt, ye = adj_true[:, j], est[:, j]
        ax.scatter(xt, ye, s=8, alpha=0.6, label=f"neuron {j}")
        lo = min(lo, xt.min(), ye.min())
        hi = max(hi, xt.max(), ye.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.5)
    ax.set(title="inferred vs true (random neurons)", xlabel="true weight", ylabel="estimated weight")
    ax.legend(fontsize=7)


# --------------------------------------------------------------------------- #
# assemble
# --------------------------------------------------------------------------- #

def render(adj_true, est, spikes, calcium, dt, title, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    ax = axes.ravel()
    _raster(ax[0], spikes, dt)
    _calcium(ax[1], calcium, spikes, dt)
    _matrix(ax[2], adj_true, "ground truth")
    _matrix(ax[3], est, "estimated")
    _three_class(ax[4], est, adj_true)
    _eigs(ax[5], est, adj_true)
    _random_neurons(ax[6], est, adj_true)
    _confusion(ax[7], est, adj_true)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def make_panel(run_dir, stage="EN+Dale", out=None):
    from calcium_ar.experiments.config import ExperimentConfig
    from calcium_ar.data.dataset import SimulatedDataset

    run_dir = Path(run_dir)
    config = ExperimentConfig.from_json(run_dir / "config.json")
    # adj_inferred uses source=column; adj_true is stored source=row. Transpose to
    # the estimated convention so every panel aligns (see metrics.py convention note).
    adj_true = np.load(run_dir / "adj_true.npy").T
    mats = {
        "EN": run_dir / "adj_inferred.npy",
        "EN+Dale": run_dir / "adj_dale.npy",
        "EN+Dale+balance": run_dir / "adj_dale_balance.npy",
    }
    est_path = mats.get(stage, mats["EN"])
    if not est_path.exists():
        est_path, stage = mats["EN"], "EN"
    est = np.load(est_path)

    ds = SimulatedDataset.load_or_generate(config, config.data_path)
    spikes = np.asarray(ds.spikes)
    calcium = np.asarray(ds.calcium)

    out = out or str(run_dir / f"panel_{stage.replace('+', '_')}.png")
    title = f"{config.name}  [{stage}]  (N={adj_true.shape[0]}, tau={config.tau}, "
    title += f"lag={config.lag_ms}ms, lam={config.lam})"
    return render(adj_true, est, spikes, calcium, config.dt, title, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="7-panel diagnostic figure for one run")
    p.add_argument("run_dir")
    p.add_argument("--stage", default="EN+Dale",
                   choices=["EN", "EN+Dale", "EN+Dale+balance"])
    p.add_argument("--out", default=None)
    a = p.parse_args()
    print("wrote", make_panel(a.run_dir, a.stage, a.out))
