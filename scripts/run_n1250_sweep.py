"""
Groups 1 & 2 solver/strength comparison at N=1250 (SLURM job array).

Each array task runs ONE solver setting on the SAME N=1250 dataset, so the runs
are directly comparable. The dataset is the one built by run_n1250.py and reused
via data_path (load_or_generate loads it instead of re-simulating).

  -> Run run_n1250.py once FIRST so the shared dataset exists. Every array task
     then just loads it (concurrent reads are fine).

Pick the task with SLURM_ARRAY_TASK_ID. Run a single one locally with:
    TASK=0 python scripts/run_n1250_sweep.py

Index -> run:
    0  OLS
    1  Ridge        lam(L2)=1e-3
    2  Lasso        lam(L1)=3e-5
    3  Lasso        lam(L1)=1e-4
    4  Lasso        lam(L1)=3e-4
    5  Lasso        lam(L1)=1e-3
    6  ElasticNet   lam(L1)=3e-5  lam_l2=1e-3
    7  ElasticNet   lam(L1)=1e-4  lam_l2=1e-3
    8  ElasticNet   lam(L1)=3e-4  lam_l2=1e-3
    9  ElasticNet   lam(L1)=1e-3  lam_l2=1e-3
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calcium_ar.experiments.config import ExperimentConfig
from calcium_ar.experiments.runner import run_single


WORKDIR = os.environ.get("CALCIUM_AR_WORKDIR", ".")

# (name, solver, lam, lam_l2)
RUNS = [
    ("ols",        "ols",           0.0,  0.0),
    ("ridge",      "ridge",         1e-3, 0.0),
    ("lasso_3e-5", "sklearn_lasso", 3e-5, 0.0),
    ("lasso_1e-4", "sklearn_lasso", 1e-4, 0.0),
    ("lasso_3e-4", "sklearn_lasso", 3e-4, 0.0),
    ("lasso_1e-3", "sklearn_lasso", 1e-3, 0.0),
    ("en_3e-5",    "fista",         3e-5, 1e-3),
    ("en_1e-4",    "fista",         1e-4, 1e-3),
    ("en_3e-4",    "fista",         3e-4, 1e-3),
    ("en_1e-3",    "fista",         1e-3, 1e-3),
]


def make_config(name, solver, lam, lam_l2):
    return ExperimentConfig(
        # --- network: same 1250-neuron net as run_n1250.py (shared dataset) ---
        n_excitatory = 1000,
        n_inhibitory = 250,
        epsilon      = 0.1,
        g            = 5.0,
        eta          = 2.0,
        J_ex         = 0.8,
        sim_time     = 50_000.0,
        dt           = 0.1,
        n_threads    = 8,            # MUST match --cpus-per-task in the SLURM script
        # --- calcium + preprocessing ---
        tau          = 100.0,
        smooth_window_ms = 3.1,
        tau_method       = "ransac",
        # --- solver (the thing we vary) ---
        solver       = solver,
        lag_ms       = 1.5,
        lam          = lam,
        lam_l2       = lam_l2,
        chunk_size   = 10_000,
        # --- data + bookkeeping ---
        data_path    = os.path.join(WORKDIR, "data", "N1250"),     # shared, built once
        output_dir   = os.path.join(WORKDIR, "results", "n1250_sweep"),
        name         = f"n1250_{name}",
        seed         = 42,
    )


if __name__ == "__main__":
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("TASK", "0")))
    if not (0 <= task < len(RUNS)):
        sys.exit(f"task index {task} out of range 0..{len(RUNS) - 1}")

    name, solver, lam, lam_l2 = RUNS[task]
    print(f"[sweep] python : {sys.executable}")
    print(f"[sweep] task {task}: {name}  solver={solver} lam={lam} lam_l2={lam_l2}")

    result = run_single(make_config(name, solver, lam, lam_l2))
    print(
        f"[sweep] DONE task {task} ({name})  "
        f"pearson={result.get('connectivity/pearson'):.4f}  "
        f"auc={result.get('connectivity/auc_roc'):.4f}  "
        f"-> {result.run_dir}"
    )
