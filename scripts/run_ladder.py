"""
Stage 2 — the regularization ladder at the operating point.

Runs OLS, Ridge, Elastic Net, and Dale-Elastic-Net on ONE fixed dataset
(operating point: lag 2 ms, tau 100, T 50k, seed 1, lam 1e-4), scores all with
the detection + 3-class metrics, saves every matrix, and draws a comparison
figure: confusion matrices + distributions per rung, and a metrics bar chart —
so you see each rung's effect on the confusion, the distribution, and the numbers.

Run as a SLURM job (it re-solves 3× + a Dale solve):
    python scripts/run_ladder.py
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # scripts/ (make_panel)

import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from calcium_ar.experiments.config import ExperimentConfig
from calcium_ar.experiments.runner import run_single
from calcium_ar.experiments.metrics import connectivity_metrics
from calcium_ar.data.dataset import SimulatedDataset
from calcium_ar.solvers import dale_fista
from calcium_ar.postprocessing import strongest_entry_types
from make_panel import _confusion, _dist


WORKDIR = os.environ.get("CALCIUM_AR_WORKDIR", ".")
SEED, LAG, TAU, T, LAM, LAM2 = 1, 2.0, 100.0, 50_000.0, 1e-4, 1e-3
METRICS = ["auc_roc", "f1", "precision", "recall", "macro_f1", "dale_type_accuracy"]
OUTDIR = os.path.join(WORKDIR, "results", "n1250_ladder")


def _cfg(solver, lam, name):
    return ExperimentConfig(
        n_excitatory=1000, n_inhibitory=250, epsilon=0.1, g=5.0, eta=2.0,
        J_ex=0.8, sim_time=T, dt=0.1, n_threads=8,
        tau=TAU, smooth_window_ms=3.1, tau_method="ransac",
        solver=solver, lag_ms=LAG, lam=lam, lam_l2=LAM2, chunk_size=10_000,
        data_path=os.path.join(WORKDIR, "data", "N1250", f"s{SEED}_tau{int(TAU)}_T{int(T)}"),
        output_dir=os.path.join(OUTDIR, name), name=f"ladder_{name}", seed=SEED,
    )


def render_ladder(rungs, adj, rows, out):
    fig = plt.figure(figsize=(20, 13))
    gs = gridspec.GridSpec(3, 4, height_ratios=[1, 1, 0.9])
    for j, (name, A) in enumerate(rungs):
        ax = fig.add_subplot(gs[0, j]); _confusion(ax, A, adj); ax.set_title(f"{name}\n(confusion)")
        ax = fig.add_subplot(gs[1, j]); _dist(ax, A, adj); ax.set_title(f"{name} (dist)")
    axm = fig.add_subplot(gs[2, :])
    names = [r["rung"] for r in rows]
    x = np.arange(len(names)); w = 0.13
    for i, k in enumerate(METRICS):
        axm.bar(x + i * w, [r[k] for r in rows], w, label=k)
    axm.set_xticks(x + w * (len(METRICS) - 1) / 2)
    axm.set_xticklabels(names)
    axm.set_ylim(0, 1.02); axm.legend(ncol=len(METRICS), fontsize=8)
    axm.set_title("metrics across the regularization ladder")
    fig.suptitle("Stage 2 — regularization ladder (operating point: lag 2 ms, lam 1e-4)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"[ladder] python: {sys.executable}")

    rungs = []
    r_ols = run_single(_cfg("ols", 0.0, "ols"))
    rungs.append(("OLS", np.load(r_ols.adj_inferred_path)))
    r_rid = run_single(_cfg("ridge", 1e-3, "ridge"))
    rungs.append(("Ridge", np.load(r_rid.adj_inferred_path)))
    r_en = run_single(_cfg("fista", LAM, "en"))
    A_en = np.load(r_en.adj_inferred_path)
    rungs.append(("EN", A_en))

    # Dale-EN: sign-constrained re-solve on the EN feed
    feed = np.asarray(zarr.open(r_en.feed_zarr_path, "r")[:])
    L = round(LAG / 0.1)
    Xc = feed[:, :-L] - feed[:, :-L].mean(1, keepdims=True)
    Yc = feed[:, L:] - feed[:, L:].mean(1, keepdims=True)
    del feed
    A_dale = dale_fista(Xc, Yc, strongest_entry_types(A_en), lam1=LAM, lam2=LAM2, n_iter=800)
    rungs.append(("Dale-EN", A_dale))

    ds = SimulatedDataset.load_or_generate(_cfg("ols", 0.0, "ols"),
                                           _cfg("ols", 0.0, "ols").data_path)
    adj_raw = np.asarray(ds.adj_true); np.fill_diagonal(adj_raw, 0.0)
    adj = adj_raw.T
    m = connectivity_metrics._metrics

    rows = []
    for name, A in rungs:
        np.save(os.path.join(OUTDIR, f"adj_{name}.npy"), A)
        row = {"rung": name, **{k: float(m[k](A, adj_raw)) for k in METRICS}}
        rows.append(row)
        print(f"  {name:8s} " + "  ".join(f"{k}={row[k]:.3f}" for k in METRICS))

    with open(os.path.join(OUTDIR, "ladder_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rung"] + METRICS)
        w.writeheader(); w.writerows(rows)

    fig = render_ladder(rungs, adj, rows, os.path.join(OUTDIR, "ladder.png"))
    print(f"[ladder] DONE -> {fig}")


if __name__ == "__main__":
    main()
