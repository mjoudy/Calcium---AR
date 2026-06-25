"""
First scale-up run on the cluster: the recommended pipeline at N=1250.

Combines the validated N=100 estimator settings (lag = 1.5 ms = synaptic delay,
FISTA Elastic Net) with a correctly-scaled 1250-neuron Brunel network
(1000 excitatory + 250 inhibitory, J_ex = 0.8 from the J_ex * C_E = 80 rule).

run_single() saves the solver result + ALL metrics to the workspace and appends
a row to the ledger. Post-processing (Dale-reg, balance rescale) is a separate
step we add after this first run works.

Usage (inside a SLURM job, env already active):
    python scripts/run_n1250.py

Big outputs go to the workspace, resolved from the CALCIUM_AR_WORKDIR env var
(set by the SLURM job script); falls back to ./results when run locally.
"""

import os
import sys

# Make the package importable no matter which folder we run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calcium_ar.experiments.config import ExperimentConfig
from calcium_ar.experiments.runner import run_single


# Where big outputs live. The job script exports CALCIUM_AR_WORKDIR=$(ws_find ...).
# Falls back to the current directory when run locally.
WORKDIR = os.environ.get("CALCIUM_AR_WORKDIR", ".")

config = ExperimentConfig(
    # --- network: 1250 neurons (1000 excitatory + 250 inhibitory) ---
    n_excitatory = 1000,
    n_inhibitory = 250,
    epsilon      = 0.1,
    g            = 5.0,
    eta          = 2.0,
    J_ex         = 0.8,          # J_ex * C_E = 80, with C_E = 0.1 * 1000 = 100
    sim_time     = 50_000.0,     # ms (starting point; we scale this up later)
    dt           = 0.1,
    n_threads    = 8,            # MUST match --cpus-per-task in the SLURM script

    # --- calcium signal ---
    tau          = 100.0,

    # --- preprocessing ---
    smooth_window_ms = 3.1,
    tau_method       = "ransac",

    # --- solver: recommended Elastic Net (FISTA) at the synaptic-delay lag ---
    solver       = "fista",
    lag_ms       = 1.5,
    lam          = 3e-3,         # L1 strength
    lam_l2       = 1e-3,         # L2 strength
    chunk_size   = 10_000,

    # --- data + bookkeeping ---
    data_path    = os.path.join(WORKDIR, "data", "N1250"),     # built once, reusable
    output_dir   = os.path.join(WORKDIR, "results", "n1250"),
    name         = "n1250_fista",
    seed         = 42,
)


if __name__ == "__main__":
    import sys
    import numpy
    import nest
    print(f"[run_n1250] python : {sys.executable}")
    print(f"[run_n1250] nest {nest.__version__} | numpy {numpy.__version__}")
    print(f"[run_n1250] workdir = {WORKDIR}")
    result = run_single(config)
    print(
        f"[run_n1250] DONE  "
        f"pearson={result.get('connectivity/pearson'):.4f}  "
        f"commit={result.git_commit[:8]}  "
        f"-> {result.run_dir}"
    )
