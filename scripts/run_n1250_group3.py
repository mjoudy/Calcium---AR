"""
Group 3 at N=1250: full recommended pipeline, scored stage by stage.

For ONE regularization strength (chosen by SLURM_ARRAY_TASK_ID) it runs:

    Elastic Net  ->  Dale re-solve (sign-constrained)  ->  balance rescale

and scores all three stages against the ground truth, so the report shows what
each step adds. Reuses the shared N=1250 dataset built by run_n1250.py.

Index -> strength:
    0  Elastic Net  lam(L1)=1e-4  (lam_l2=1e-3)
    1  Elastic Net  lam(L1)=3e-4  (lam_l2=1e-3)

Run one locally:  TASK=0 python scripts/run_n1250_group3.py
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import zarr

from calcium_ar.experiments.config import ExperimentConfig
from calcium_ar.experiments.runner import run_single
from calcium_ar.experiments.metrics import connectivity_metrics
from calcium_ar.data.dataset import SimulatedDataset
from calcium_ar.solvers import dale_fista
from calcium_ar.postprocessing import (
    strongest_entry_types,
    rescale_balance_nz,
)


WORKDIR = os.environ.get("CALCIUM_AR_WORKDIR", ".")

# (name, lam_l1, lam_l2)
RUNS = [
    ("en_1e-4", 1e-4, 1e-3),
    ("en_3e-4", 3e-4, 1e-3),
]

REPORT_METRICS = [
    "pearson", "spearman", "auc_roc", "f1",
    "precision", "recall", "dale_type_accuracy", "degree_of_daleianity",
]


def make_config(name, lam, lam_l2):
    return ExperimentConfig(
        n_excitatory=1000, n_inhibitory=250, epsilon=0.1, g=5.0, eta=2.0,
        J_ex=0.8, sim_time=50_000.0, dt=0.1, n_threads=8,
        tau=100.0, smooth_window_ms=3.1, tau_method="ransac",
        solver="fista", lag_ms=1.5, lam=lam, lam_l2=lam_l2, chunk_size=10_000,
        data_path=os.path.join(WORKDIR, "data", "N1250"),       # shared dataset
        output_dir=os.path.join(WORKDIR, "results", "n1250_group3"),
        name=f"n1250_group3_{name}", seed=42,
    )


def ei_ratio(A, adj):
    """Median |inhibitory| / median |excitatory| weight (true 5.0 for this net)."""
    off = ~np.eye(A.shape[0], dtype=bool)
    g = adj.T[off]
    a = np.abs(A[off])
    inh, exc = a[g < 0], a[g > 0]
    if len(inh) and len(exc) and np.median(exc) > 0:
        return float(np.median(inh) / np.median(exc))
    return float("nan")


def score(A, adj, m):
    A = A.copy()
    np.fill_diagonal(A, 0.0)
    row = {k: float(m[k](A, adj)) for k in REPORT_METRICS}
    row["ei_ratio"] = ei_ratio(A, adj)
    return row


if __name__ == "__main__":
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("TASK", "0")))
    if not (0 <= task < len(RUNS)):
        sys.exit(f"task index {task} out of range 0..{len(RUNS) - 1}")

    name, lam, lam_l2 = RUNS[task]
    config = make_config(name, lam, lam_l2)
    print(f"[group3] python : {sys.executable}")
    print(f"[group3] task {task}: {name}  lam={lam} lam_l2={lam_l2}")

    # 1. Elastic Net via run_single (also logs to the main ledger + saves the feed).
    result = run_single(config)
    A_en = np.load(result.adj_inferred_path)

    # 2. Load the preprocessed feed, ground truth, and firing rates.
    feed = np.asarray(zarr.open(result.feed_zarr_path, "r")[:])
    ds = SimulatedDataset.load_or_generate(config, config.data_path)
    adj = np.asarray(ds.adj_true)
    np.fill_diagonal(adj, 0.0)
    rates = np.asarray(ds.spikes).sum(1) / (config.sim_time / 1000.0)   # Hz per neuron

    # 3. Dale re-solve (sign-constrained EN) on centred lag pairs.
    LAG = round(config.lag_ms / config.dt)
    Xc = feed[:, :-LAG] - feed[:, :-LAG].mean(1, keepdims=True)
    Yc = feed[:, LAG:] - feed[:, LAG:].mean(1, keepdims=True)
    del feed
    t_unsup = strongest_entry_types(A_en)
    A_dale = dale_fista(Xc, Yc, t_unsup, lam1=lam, lam2=lam_l2, n_iter=800)

    # 4. Balance rescale on top.
    A_full = rescale_balance_nz(A_dale, rates)

    # 5. Score every stage and write a per-strength CSV for the report.
    m = connectivity_metrics._metrics
    stages = [("EN", A_en), ("EN+Dale", A_dale), ("EN+Dale+balance", A_full)]

    os.makedirs(config.output_dir, exist_ok=True)
    out_csv = os.path.join(config.output_dir, f"group3_{name}.csv")
    cols = ["lam", "stage"] + REPORT_METRICS + ["ei_ratio"]

    print(f"\n[group3] {name} (lam={lam}) — stage scores:")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for stage_name, A in stages:
            s = score(A, adj, m)
            w.writerow({"lam": lam, "stage": stage_name, **s})
            print(f"  {stage_name:18s} pearson={s['pearson']:.3f} "
                  f"auc={s['auc_roc']:.3f} type_acc={s['dale_type_accuracy']:.3f} "
                  f"ei={s['ei_ratio']:.2f}")

    # Keep the post-processed matrices next to the EN run for later analysis.
    np.save(os.path.join(result.run_dir, "adj_dale.npy"), A_dale)
    np.save(os.path.join(result.run_dir, "adj_dale_balance.npy"), A_full)

    print(f"[group3] DONE {name} -> {out_csv}")
