"""
ExperimentConfig — single source of truth for every tunable parameter
across the full pipeline: simulation → calcium → preprocessing → solver.

Serialises to / from plain JSON so configs are human-readable and
reproducible on both laptop and HPC.
"""

from __future__ import annotations
import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class ExperimentConfig:
    # ------------------------------------------------------------------ #
    # Brunel network
    # ------------------------------------------------------------------ #
    n_excitatory: int   = 800
    n_inhibitory: int   = 200
    sim_time: float     = 5_000.0   # ms
    dt: float           = 0.1       # ms  (time resolution)
    epsilon: float      = 0.1       # connection probability
    g: float            = 5.0       # |J_in| / J_ex ratio
    eta: float          = 2.0       # external drive relative to threshold
    # J_ex scaling rule: J_ex × C_E = 80  (Brunel 2000: C_E=800, J=0.1)
    # NE=800 → C_E=80 → J_ex = 80/80 = 1.0 mV
    # NE=1000 → C_E=100 → J_ex = 80/100 = 0.8 mV
    # NE=80  → C_E=8   → J_ex = 80/8  = 10.0 mV  (few-synapse regime)
    J_ex: float         = 1.0       # mV  (correct for NE=800 default)
    n_threads: int      = 1         # NEST CPU threads

    # ------------------------------------------------------------------ #
    # Calcium signal
    # ------------------------------------------------------------------ #
    tau: float          = 100.0     # true decay time constant (ms)
    amplitude: float    = 1.0       # ΔF per spike
    sigma_intra: float  = 0.01      # intracellular noise std
    sigma_extra: float  = 0.05      # recording noise std

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #
    smooth_window_ms: float = 3.1   # Savitzky-Golay window duration (ms); converted
                                    # to samples as round(smooth_window_ms / dt),
                                    # enforced odd and >= 5
    tau_method: str     = "ransac"  # ols | ransac | dbscan | iqr | zscore
    spike_cut_window: int = 5       # half-window around spikes to remove

    # ------------------------------------------------------------------ #
    # Solver
    # ------------------------------------------------------------------ #
    solver: str         = "ridge"   # ols | ridge | fista |
                                    # torch_normal_eq | torch_minibatch |
                                    # torch_linear_layer | sklearn_lasso
    lag_ms: float       = 10.0      # AR lag in physical time (ms); converted to
                                    # samples as round(lag_ms / dt) by runner
    lam: float          = 1.0       # regularisation strength (L1 for fista/lasso,
                                    # L2 for ridge)
    lam_l2: float       = 1e-3      # L2 strength for fista (Elastic Net only)
    chunk_size: int     = 10_000    # time steps per chunk / mini-batch
    n_epochs: int       = 10        # gradient-descent epochs (torch only)
    lr: float           = 1e-3      # Adam learning rate (torch only)
    device: str | None  = None      # 'cuda' | 'cpu' | None (auto)

    # ------------------------------------------------------------------ #
    # Data management
    # ------------------------------------------------------------------ #
    # None  → simulate fresh every run (don't save)
    # path  → use SimulatedDataset.load_or_generate():
    #          load if the dataset already exists, generate + save otherwise.
    #          Ideal for sweeps: build the dataset once, reuse it many times.
    data_path: str | None = None

    # ------------------------------------------------------------------ #
    # Run metadata
    # ------------------------------------------------------------------ #
    name: str           = "experiment"
    seed: int           = 42
    output_dir: str     = "results"

    # ------------------------------------------------------------------ #
    # Serialisation helpers
    # ------------------------------------------------------------------ #

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(dataclasses.asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def replace(self, **changes) -> ExperimentConfig:
        """Return a copy with selected fields overridden."""
        return dataclasses.replace(self, **changes)
