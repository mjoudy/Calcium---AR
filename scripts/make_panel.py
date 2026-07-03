"""
Compact diagnostic panel for one estimation.

Six cells: ground truth, estimated matrix, 3-class confusion (row-normalized),
estimated-weight distribution by true class, a precision-recall curve, and a box
of the key numbers.

Note on orientation: adj_inferred[i,j] <-> adj_true.T[i,j] (see metrics.py). So the
ground truth is transposed to the estimate's convention for the visuals, while the
metric functions get the raw adj_true (they transpose internally).

_raster / _calcium are kept here because scripts/network_stats.py imports them.

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
# network-signal helpers (used by scripts/network_stats.py)
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


# --------------------------------------------------------------------------- #
# panel pieces
# --------------------------------------------------------------------------- #

def _matrix(ax, A, title):
    nz = np.abs(A[A != 0])
    v = np.percentile(nz, 99) if nz.size else 1.0
    im = ax.imshow(A, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set(title=title, xlabel="source", ylabel="target")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def _dist(ax, est, adj):
    """Estimated-weight distribution split by TRUE class. Each class normalized to
    its own area (density) so shapes are comparable despite the huge size gap;
    log-y keeps the small E/I tails visible under the giant 'none' peak."""
    off = ~np.eye(adj.shape[0], dtype=bool)
    g, e = adj[off], est[off]
    for mask, lab, c in [(g > 0, "true E", "tab:red"),
                         (g == 0, "true none", "0.5"),
                         (g < 0, "true I", "tab:blue")]:
        v = e[mask]
        if v.size:
            ax.hist(v, bins=80, density=True, histtype="step", lw=1.6, label=lab, color=c)
    ax.set(title="estimated weight by true class (density)",
           xlabel="estimated weight", ylabel="density per class (log)")
    ax.set_yscale("log")
    ax.legend(fontsize=7)


def _confusion(ax, est, adj, eps=1e-9):
    """3-class confusion (E / none / I), coloured by ROW-normalized rate (recall)
    so it is comparable across configs. Cells show count + row %."""
    off = ~np.eye(adj.shape[0], dtype=bool)

    def lab(A):
        s = np.sign(A[off]).astype(int)
        s[np.abs(A[off]) <= eps] = 0
        return s

    t, p = lab(adj), lab(est)
    classes, names = [1, 0, -1], ["E", "none", "I"]
    M = np.array([[int(np.sum((t == ct) & (p == cp))) for cp in classes]
                  for ct in classes])
    R = M / np.maximum(M.sum(1, keepdims=True), 1)
    ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=[0, 1, 2], yticks=[0, 1, 2], xticklabels=names, yticklabels=names,
           xlabel="predicted", ylabel="true", title="confusion (row-normalized = recall)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:,}\n{R[i, j] * 100:.1f}%", ha="center", va="center",
                    fontsize=8, color="white" if R[i, j] > 0.5 else "black")


def _pr_curve(ax, est, adj, max_pts=3_000_000):
    """Precision-recall for 'does a connection exist' (score = |estimate|).
    PR is more honest than ROC here because 'none' hugely outnumbers connections."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    off = ~np.eye(adj.shape[0], dtype=bool)
    y = (adj[off] != 0).astype(int)
    s = np.abs(est[off])
    if y.size > max_pts:                       # subsample for very large N
        idx = np.random.default_rng(0).choice(y.size, max_pts, replace=False)
        y, s = y[idx], s[idx]
    pr, rc, _ = precision_recall_curve(y, s)
    ap = average_precision_score(y, s)
    ax.plot(rc, pr, lw=1.6)
    ax.axhline(y.mean(), ls="--", c="0.6", lw=0.8, label=f"chance = {y.mean():.3f}")
    ax.set(title=f"precision–recall  (AP = {ap:.3f})",
           xlabel="recall", ylabel="precision", ylim=(0, 1.02))
    ax.legend(fontsize=7)


def _metrics_box(ax, metrics):
    ax.axis("off")
    lines = [f"{k:>16s} : {v:.3f}" for k, v in metrics.items()]
    ax.text(0.02, 0.97, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=11, transform=ax.transAxes)
    ax.set_title("metrics")


# --------------------------------------------------------------------------- #
# assemble
# --------------------------------------------------------------------------- #

REPORT = ["pearson", "spearman", "auc_roc", "f1",
          "precision", "recall", "macro_f1", "dale_type_accuracy"]


def render(adj, est, metrics, title, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax = axes.ravel()
    _matrix(ax[0], adj, "ground truth")
    _matrix(ax[1], est, "estimated")
    _confusion(ax[2], est, adj)
    _dist(ax[3], est, adj)
    _pr_curve(ax[4], est, adj)
    _metrics_box(ax[5], metrics)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def make_panel(run_dir, stage="EN+Dale", out=None):
    from calcium_ar.experiments.config import ExperimentConfig
    from calcium_ar.experiments.metrics import connectivity_metrics

    run_dir = Path(run_dir)
    config = ExperimentConfig.from_json(run_dir / "config.json")
    adj_raw = np.load(run_dir / "adj_true.npy")   # source=row; metrics transpose internally
    adj = adj_raw.T                               # aligned to the estimate for the visuals

    mats = {
        "EN": run_dir / "adj_inferred.npy",
        "EN+Dale": run_dir / "adj_dale.npy",
        "EN+Dale+balance": run_dir / "adj_dale_balance.npy",
    }
    est_path = mats.get(stage, mats["EN"])
    if not est_path.exists():
        est_path, stage = mats["EN"], "EN"
    est = np.load(est_path)

    m = connectivity_metrics._metrics
    metrics = {k: float(m[k](est, adj_raw)) for k in REPORT}

    out = out or str(run_dir / f"panel_{stage.replace('+', '_')}.png")
    title = (f"{config.name}  [{stage}]   N={adj.shape[0]}, tau={config.tau}, "
             f"lag={config.lag_ms} ms, lam={config.lam}")
    return render(adj, est, metrics, title, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="compact diagnostic panel for one run")
    p.add_argument("run_dir")
    p.add_argument("--stage", default="EN+Dale",
                   choices=["EN", "EN+Dale", "EN+Dale+balance"])
    p.add_argument("--out", default=None)
    a = p.parse_args()
    print("wrote", make_panel(a.run_dir, a.stage, a.out))
