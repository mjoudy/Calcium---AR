"""
Stage 3 — detection decision analysis for one estimation.

Treats |estimated weight| as a detection score for "does a connection exist", and:
  - draws the ROC and precision-recall curves (AUC, AP),
  - sweeps the threshold and plots precision / recall / F1 vs threshold,
  - marks TWO thresholds:
      * ORACLE   = the threshold that maximizes F1 (uses ground truth),
      * GT-FREE  = an unsupervised threshold from a 2-component mixture on the score
                   distribution (no ground truth) — the real-data-usable choice.
  - prints precision/recall/F1 at each, and the F1 gap between them.

The key question it answers: does the unsupervised threshold match the oracle?
If yes, the method is usable on experimental data.

Usage: python scripts/decision_analysis.py <run_dir> [--stage ...] [--out fig.png]
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


def gmm_threshold(scores):
    """Unsupervised detection threshold: 2-component Gaussian mixture on the
    non-zero |scores|; threshold = where the 'signal' component overtakes 'noise'."""
    from sklearn.mixture import GaussianMixture
    x = scores[scores > 0]
    if x.size < 50:
        return float(np.median(scores))
    xs = x if x.size < 200_000 else np.random.default_rng(0).choice(x, 200_000, replace=False)
    g = GaussianMixture(2, random_state=0, n_init=2).fit(xs.reshape(-1, 1))
    means = g.means_.ravel()
    hi = int(np.argmax(means))
    grid = np.linspace(means.min(), means.max(), 1000).reshape(-1, 1)
    post = g.predict_proba(grid)[:, hi]
    idx = np.argmax(post > 0.5)
    return float(grid[idx, 0]) if np.any(post > 0.5) else float(means.max())


def _pr_at(y, s, thr):
    pred = (s >= thr)
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def analyse(adj, est, max_pts=4_000_000):
    from sklearn.metrics import (roc_curve, roc_auc_score,
                                 precision_recall_curve, average_precision_score)
    off = ~np.eye(adj.shape[0], dtype=bool)
    y = (adj[off] != 0).astype(int)
    s = np.abs(est[off])
    if y.size > max_pts:
        idx = np.random.default_rng(0).choice(y.size, max_pts, replace=False)
        y, s = y[idx], s[idx]

    fpr, tpr, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    prec, rec, thr = precision_recall_curve(y, s)
    ap = average_precision_score(y, s)

    # F1 vs threshold from the PR curve (prec/rec have len n+1, thr has len n)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    j = int(np.argmax(f1[:-1]))
    oracle_thr = float(thr[j])
    gtfree_thr = gmm_threshold(s)

    return dict(y=y, s=s, fpr=fpr, tpr=tpr, auc=auc, prec=prec, rec=rec, thr=thr,
                ap=ap, oracle_thr=oracle_thr, gtfree_thr=gtfree_thr)


def render(a, title, out):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))

    ax[0].plot(a["fpr"], a["tpr"], lw=1.6)
    ax[0].plot([0, 1], [0, 1], "--", c="0.6", lw=0.8)
    ax[0].set(title=f"ROC (AUC = {a['auc']:.3f})", xlabel="false positive rate",
              ylabel="true positive rate", xlim=(0, 1), ylim=(0, 1.02))

    ax[1].plot(a["rec"], a["prec"], lw=1.6)
    ax[1].axhline(a["y"].mean(), ls="--", c="0.6", lw=0.8, label=f"chance = {a['y'].mean():.3f}")
    ax[1].set(title=f"precision–recall (AP = {a['ap']:.3f})", xlabel="recall",
              ylabel="precision", xlim=(0, 1), ylim=(0, 1.02))
    ax[1].legend(fontsize=8)

    # precision / recall / F1 vs threshold
    t = a["thr"]
    ax[2].plot(t, a["prec"][:-1], label="precision", lw=1.4)
    ax[2].plot(t, a["rec"][:-1], label="recall", lw=1.4)
    f1 = 2 * a["prec"] * a["rec"] / (a["prec"] + a["rec"] + 1e-12)
    ax[2].plot(t, f1[:-1], label="F1", lw=1.4, color="k")
    for thr, name, c in [(a["oracle_thr"], "oracle (max-F1)", "tab:green"),
                         (a["gtfree_thr"], "GT-free (mixture)", "tab:red")]:
        ax[2].axvline(thr, ls="--", c=c, lw=1.2, label=f"{name} = {thr:.4f}")
    ax[2].set(title="precision / recall / F1 vs threshold", xlabel="threshold on |weight|",
              ylabel="score", ylim=(0, 1.02))
    ax[2].set_xscale("log")
    ax[2].legend(fontsize=7)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def decision_analysis(run_dir, stage="EN+Dale", out=None):
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

    a = analyse(adj, est)
    op = _pr_at(a["y"], a["s"], a["oracle_thr"])
    gf = _pr_at(a["y"], a["s"], a["gtfree_thr"])
    print(f"[decision] {config.name} [{stage}]  AUC={a['auc']:.3f}  AP={a['ap']:.3f}")
    print(f"  oracle  thr={a['oracle_thr']:.4f}  P={op[0]:.3f} R={op[1]:.3f} F1={op[2]:.3f}")
    print(f"  GT-free thr={a['gtfree_thr']:.4f}  P={gf[0]:.3f} R={gf[1]:.3f} F1={gf[2]:.3f}")
    print(f"  F1 gap (oracle - GT-free) = {op[2] - gf[2]:.3f}")

    out = out or str(run_dir / f"decision_{stage.replace('+', '_')}.png")
    title = (f"{config.name}  [{stage}]   AUC={a['auc']:.3f}  AP={a['ap']:.3f}  "
             f"| F1: oracle={op[2]:.3f}, GT-free={gf[2]:.3f}")
    return render(a, title, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="detection ROC/PR + oracle vs GT-free threshold")
    p.add_argument("run_dir")
    p.add_argument("--stage", default="EN+Dale",
                   choices=["EN", "EN+Dale", "EN+Dale+balance"])
    p.add_argument("--out", default=None)
    a = p.parse_args()
    print("wrote", decision_analysis(a.run_dir, a.stage, a.out))
