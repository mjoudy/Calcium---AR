"""
Parameter sweeps at N=1250: EN -> Dale -> balance, every stage scored, matrices saved.

Which sweep runs is chosen by the SWEEP env var; which point by SLURM_ARRAY_TASK_ID.
A special SWEEP=prep generates all datasets first (one per task, unique paths) so
the sweep tasks only LOAD them — no concurrent-generation races.

Reference network (N=1250, recommended): 1000 exc + 250 inh, g=5, J_ex=0.8, eta=2,
eps=0.1, tau=100 ms, T=50k ms, lag=1.5 ms, Elastic Net lam=1e-4 / l2=1e-3.
Each sweep varies ONE knob across SEEDS seeds. Datasets are keyed by
(seed, tau, sim_time), so the lag and lam sweeps reuse the reference datasets.

Submit (exact commands in slurm/run_experiments.slurm):
    SWEEP=prep  array 0-54   # run FIRST, wait for it to finish
    SWEEP=lag   array 0-59
    SWEEP=lam   array 0-29
    SWEEP=tau   array 0-34
    SWEEP=data  array 0-24
Run one locally:  SWEEP=lag TASK=0 python scripts/run_n1250_experiments.py
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
from calcium_ar.postprocessing import strongest_entry_types, rescale_balance_nz


WORKDIR = os.environ.get("CALCIUM_AR_WORKDIR", ".")
SEEDS = [1, 2, 3, 4, 5]   # NEST requires seed >= 1 (seed 0 is invalid)

# reference operating point (the fixed knobs; each sweep moves one of them)
REF_TAU, REF_T, REF_LAG, REF_LAM = 100.0, 50_000.0, 1.5, 1e-4

# sweep name -> (ExperimentConfig field, values to try)
SWEEPS = {
    "lag":  ("lag_ms",   [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]),
    "lam":  ("lam",      [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]),
    "tau":  ("tau",      [1.0, 10.0, 50.0, 100.0, 200.0, 400.0, 800.0]),
    "data": ("sim_time", [10_000.0, 25_000.0, 50_000.0, 100_000.0, 200_000.0]),
}

REPORT_METRICS = [
    "pearson", "spearman", "auc_roc", "f1",
    "precision", "recall", "dale_type_accuracy", "degree_of_daleianity",
]


def _fmt(v):
    return repr(v)


def _data_path(seed, tau, sim_time):
    return os.path.join(WORKDIR, "data", "N1250", f"s{seed}_tau{int(tau)}_T{int(sim_time)}")


def _config(seed, tau, sim_time, lag, lam, name, out_sub):
    return ExperimentConfig(
        n_excitatory=1000, n_inhibitory=250, epsilon=0.1, g=5.0, eta=2.0,
        J_ex=0.8, sim_time=sim_time, dt=0.1, n_threads=8,
        tau=tau, smooth_window_ms=3.1, tau_method="ransac",
        solver="fista", lag_ms=lag, lam=lam, lam_l2=1e-3, chunk_size=10_000,
        data_path=_data_path(seed, tau, sim_time),
        output_dir=os.path.join(WORKDIR, "results", "n1250_sweeps", out_sub),
        name=name, seed=seed,
    )


def build_config(sweep, value, seed):
    tau, sim_time, lag, lam = REF_TAU, REF_T, REF_LAG, REF_LAM
    if   sweep == "tau":  tau = value
    elif sweep == "data": sim_time = value
    elif sweep == "lag":  lag = value
    elif sweep == "lam":  lam = value
    tag = f"{sweep}_{_fmt(value)}_s{seed}"
    # per-task output dir -> per-task ledger (no concurrent-write races)
    return _config(seed, tau, sim_time, lag, lam,
                   name=f"n1250_{tag}", out_sub=os.path.join(sweep, tag))


def dataset_config(seed, tau, sim_time):
    return _config(seed, tau, sim_time, REF_LAG, REF_LAM,
                   name=f"prep_s{seed}_tau{int(tau)}_T{int(sim_time)}", out_sub="_prep")


def sweep_grid(sweep):
    _, values = SWEEPS[sweep]
    return [(v, s) for v in values for s in SEEDS]


def unique_datasets():
    """Every distinct (seed, tau, sim_time) any sweep needs."""
    combos = set()
    for s in SEEDS:
        combos.add((s, REF_TAU, REF_T))                 # reference (lag & lam sweeps)
        for tau in SWEEPS["tau"][1]:
            combos.add((s, tau, REF_T))                 # tau sweep
        for T in SWEEPS["data"][1]:
            combos.add((s, REF_TAU, T))                 # data sweep
    return sorted(combos)


def _ei_ratio(A, adj):
    off = ~np.eye(A.shape[0], dtype=bool)
    g = adj.T[off]; a = np.abs(A[off])
    inh, exc = a[g < 0], a[g > 0]
    if len(inh) and len(exc) and np.median(exc) > 0:
        return float(np.median(inh) / np.median(exc))
    return float("nan")


def _score(A, adj, m):
    A = A.copy(); np.fill_diagonal(A, 0.0)
    row = {k: float(m[k](A, adj)) for k in REPORT_METRICS}
    row["ei_ratio"] = _ei_ratio(A, adj)
    return row


def run_sweep_point(sweep, value, seed):
    cfg = build_config(sweep, value, seed)
    field = SWEEPS[sweep][0]
    if not os.path.exists(cfg.data_path):
        sys.exit(f"dataset missing: {cfg.data_path}\n-> run SWEEP=prep first.")
    print(f"[exp] {sweep}: {field}={value} seed={seed}")

    result = run_single(cfg)                 # EN: saves feed + adj_inferred + per-task ledger
    A_en = np.load(result.adj_inferred_path)

    feed = np.asarray(zarr.open(result.feed_zarr_path, "r")[:])
    ds = SimulatedDataset.load_or_generate(cfg, cfg.data_path)
    adj = np.asarray(ds.adj_true); np.fill_diagonal(adj, 0.0)
    rates = np.asarray(ds.spikes).sum(1) / (cfg.sim_time / 1000.0)

    LAG = round(cfg.lag_ms / cfg.dt)
    Xc = feed[:, :-LAG] - feed[:, :-LAG].mean(1, keepdims=True)
    Yc = feed[:, LAG:] - feed[:, LAG:].mean(1, keepdims=True)
    del feed
    t = strongest_entry_types(A_en)
    A_dale = dale_fista(Xc, Yc, t, lam1=cfg.lam, lam2=cfg.lam_l2, n_iter=800)
    A_full = rescale_balance_nz(A_dale, rates)
    np.save(os.path.join(result.run_dir, "adj_dale.npy"), A_dale)
    np.save(os.path.join(result.run_dir, "adj_dale_balance.npy"), A_full)

    m = connectivity_metrics._metrics
    stages = [("EN", A_en), ("EN+Dale", A_dale), ("EN+Dale+balance", A_full)]

    out_csv = os.path.join(cfg.output_dir, "stages.csv")
    cols = ["sweep", "param", "value", "seed", "stage"] + REPORT_METRICS + ["ei_ratio"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for stage, A in stages:
            s = _score(A, adj, m)
            w.writerow({"sweep": sweep, "param": field, "value": value,
                        "seed": seed, "stage": stage, **s})
            print(f"  {stage:18s} pearson={s['pearson']:.3f} auc={s['auc_roc']:.3f} "
                  f"type_acc={s['dale_type_accuracy']:.3f} ei={s['ei_ratio']:.2f}")
    print(f"[exp] DONE {sweep} {field}={value} seed={seed} -> {out_csv}")


if __name__ == "__main__":
    sweep = os.environ.get("SWEEP", "")
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("TASK", "0")))
    print(f"[exp] python: {sys.executable}  SWEEP={sweep} TASK={task}")

    if sweep == "prep":
        combos = unique_datasets()
        if not (0 <= task < len(combos)):
            sys.exit(f"prep task {task} out of range 0..{len(combos) - 1}")
        seed, tau, sim_time = combos[task]
        cfg = dataset_config(seed, tau, sim_time)
        print(f"[exp] prep {task}: seed={seed} tau={tau} T={sim_time} -> {cfg.data_path}")
        SimulatedDataset.load_or_generate(cfg, cfg.data_path)
        print(f"[exp] prep {task} DONE")
        sys.exit(0)

    if sweep not in SWEEPS:
        sys.exit(f"set SWEEP to one of {list(SWEEPS)} or 'prep' (got '{sweep}')")
    grid = sweep_grid(sweep)
    if not (0 <= task < len(grid)):
        sys.exit(f"{sweep} task {task} out of range 0..{len(grid) - 1}")
    value, seed = grid[task]
    run_sweep_point(sweep, value, seed)
