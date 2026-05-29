"""
ExperimentResult — everything produced by a single run.

metrics is a flat dict keyed by  <domain>/<metric_name>  (e.g.
'connectivity/pearson', 'tau/tau_mae', 'solver/final_loss').
New metrics registered in metrics.py appear here automatically.
"""

from __future__ import annotations
import dataclasses
from typing import Dict, List


@dataclasses.dataclass
class ExperimentResult:
    # ------------------------------------------------------------------ #
    # Back-reference to config
    # ------------------------------------------------------------------ #
    config_path: str                        # absolute path to config.json

    # ------------------------------------------------------------------ #
    # All evaluation metrics — keyed by "<domain>/<name>"
    # e.g. connectivity/pearson, tau/tau_mae, solver/final_loss
    # ------------------------------------------------------------------ #
    metrics: Dict[str, float]

    # ------------------------------------------------------------------ #
    # Solver convergence curve (empty for non-gradient solvers)
    # ------------------------------------------------------------------ #
    loss_curve: List[float]

    # ------------------------------------------------------------------ #
    # Timing
    # ------------------------------------------------------------------ #
    duration_seconds: float
    timestamp: str                          # ISO-8601

    # ------------------------------------------------------------------ #
    # Artifact paths (absolute)
    # ------------------------------------------------------------------ #
    run_dir: str
    spikes_path: str | None        # None when data lives in the dataset zarr
    calcium_path: str | None       # None when data lives in the dataset zarr
    feed_zarr_path: str
    adj_true_path: str
    adj_inferred_path: str

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #

    def get(self, key: str, default=float("nan")) -> float:
        """Look up a metric by its full key, e.g. 'connectivity/pearson'."""
        return self.metrics.get(key, default)

    def summary(self) -> dict:
        """Flat dict ready for a DataFrame row (metrics + timing + run_dir)."""
        row = dict(self.metrics)
        row["duration_s"] = self.duration_seconds
        row["timestamp"]  = self.timestamp
        row["run_dir"]    = self.run_dir
        return row
