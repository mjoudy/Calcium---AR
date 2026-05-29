"""
Saving and loading experiment runs.

Directory layout for one run
-----------------------------
{output_dir}/{name}_{timestamp}/
    config.json
    metrics.json
    loss_curve.npy      (empty array for non-gradient solvers)
    spikes.npy          (N, T) binary
    calcium.npy         (N, T) fluorescence
    adj_true.npy        (N, N) ground truth
    adj_inferred.npy    (N, N) estimated by solver
    feed.zarr/          (N, T) preprocessed feed (zarr array)
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import zarr

from .config import ExperimentConfig
from .result import ExperimentResult


# --------------------------------------------------------------------------- #
# Internal JSON encoder — handles numpy scalars transparently
# --------------------------------------------------------------------------- #

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# --------------------------------------------------------------------------- #
# Run-directory helpers
# --------------------------------------------------------------------------- #

def make_run_dir(config: ExperimentConfig) -> Path:
    """Create a timestamped directory for one run and return its path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    dirname = f"{config.name}_{ts}_{uid}"
    run_dir = Path(config.output_dir) / dirname
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_feed_zarr(feed: np.ndarray, run_dir: Path) -> str:
    """Persist the preprocessed feed as a zarr array; return absolute path."""
    zarr_path = str(run_dir / "feed.zarr")
    chunk_t = min(10_000, feed.shape[1])
    z = zarr.open_array(
        zarr_path,
        mode="w",
        shape=feed.shape,
        dtype=feed.dtype,
        chunks=(feed.shape[0], chunk_t),
    )
    z[:] = feed
    return zarr_path


# --------------------------------------------------------------------------- #
# Save a completed result
# --------------------------------------------------------------------------- #

def save_result(
    result: ExperimentResult,
    config: ExperimentConfig,
    spikes: np.ndarray | None,
    calcium: np.ndarray | None,
    adj_true: np.ndarray,
    adj_inferred: np.ndarray,
) -> None:
    """Write all arrays and metadata to result.run_dir.

    spikes and calcium may be None in the streaming path — the raw data
    already lives in the dataset zarr pointed to by config.data_path.
    """
    run_dir = Path(result.run_dir)

    # Config
    config.to_json(run_dir / "config.json")

    # All metrics (flat dict) + run metadata
    metrics_payload = dict(result.metrics)
    metrics_payload["duration_s"] = result.duration_seconds
    metrics_payload["timestamp"]  = result.timestamp
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, cls=_NumpyEncoder)
    )

    # Convergence curve
    np.save(run_dir / "loss_curve.npy", np.array(result.loss_curve))

    # Arrays — skip spikes/calcium when they live in the dataset zarr
    if spikes is not None:
        np.save(run_dir / "spikes.npy", spikes)
    if calcium is not None:
        np.save(run_dir / "calcium.npy", calcium)
    np.save(run_dir / "adj_true.npy",     adj_true)
    np.save(run_dir / "adj_inferred.npy", adj_inferred)


# --------------------------------------------------------------------------- #
# Load a completed result
# --------------------------------------------------------------------------- #

def load_result(run_dir: str | Path) -> tuple[ExperimentResult, ExperimentConfig]:
    """Reconstruct (ExperimentResult, ExperimentConfig) from a saved run dir."""
    run_dir = Path(run_dir)

    config = ExperimentConfig.from_json(run_dir / "config.json")
    payload    = json.loads((run_dir / "metrics.json").read_text())
    loss_curve = np.load(run_dir / "loss_curve.npy").tolist()

    # Separate run-meta keys from metric keys
    duration  = payload.pop("duration_s")
    timestamp = payload.pop("timestamp")

    result = ExperimentResult(
        config_path      = str(run_dir / "config.json"),
        metrics          = payload,        # everything that remains is a metric
        loss_curve       = loss_curve,
        duration_seconds = duration,
        timestamp        = timestamp,
        run_dir          = str(run_dir),
        spikes_path      = str(run_dir / "spikes.npy"),
        calcium_path     = str(run_dir / "calcium.npy"),
        feed_zarr_path   = str(run_dir / "feed.zarr"),
        adj_true_path    = str(run_dir / "adj_true.npy"),
        adj_inferred_path= str(run_dir / "adj_inferred.npy"),
    )
    return result, config


# --------------------------------------------------------------------------- #
# Aggregate a sweep into a DataFrame
# --------------------------------------------------------------------------- #

def load_sweep_results(sweep_dir: str | Path):
    """
    Load all run sub-directories inside sweep_dir and return a pandas DataFrame.

    Each row = one run. Columns = all config fields + metric fields.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required to aggregate sweep results: pip install pandas")

    sweep_dir = Path(sweep_dir)
    rows = []
    for run_dir in sorted(sweep_dir.iterdir()):
        if not (run_dir / "config.json").exists():
            continue
        result, config = load_result(run_dir)
        row = dataclasses.asdict(config)
        row.update(result.summary())
        rows.append(row)

    return pd.DataFrame(rows)
