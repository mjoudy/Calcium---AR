"""
run_single — orchestrates the full pipeline for one ExperimentConfig.

Stages
------
1. Brunel network simulation  (NEST)
2. Calcium signal generation  (AR1 model)
3. Preprocessing              (smooth → tau estimation → feed reconstruction)
4. Connectivity inference     (chosen solver)
5. Evaluation                 (all metrics registered in metrics.py)
6. Persistence                (all arrays + metrics saved to disk)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from ..data.dataset import SimulatedDataset
from ..simulation.brunel_network import BrunelNetwork
from ..simulation.calcium_signal import simulate_calcium
from ..preprocessing.signal_utils import get_signal_derivative_pair
from ..preprocessing.tau_estimation import estimate_tau_robust
from ..preprocessing.feed_reconstruction import reconstruct_feed, stream_feed_to_zarr
from .. import solvers as _slv
from .config import ExperimentConfig
from .result import ExperimentResult
from .persistence import make_run_dir, save_feed_zarr, save_result
from .metrics import compute_all as compute_metrics


# --------------------------------------------------------------------------- #
# Solver dispatch
# --------------------------------------------------------------------------- #

_SOLVER_MAP = {
    "ols":                "solve_ols",
    "sklearn_ols":        "solve_sklearn_ols",
    "ridge":              "solve_ridge",
    "torch_normal_eq":    "solve_torch_normal_eq",
    "torch_minibatch":    "solve_torch_minibatch",
    "torch_linear_layer": "solve_torch_linear_layer",
    "torch_gd":           "solve_torch_gd",
    "sklearn_lasso":      "solve_sklearn_lasso",
    "fista":              "solve_fista",
}


# --------------------------------------------------------------------------- #
# Data loading / simulation
# --------------------------------------------------------------------------- #

def _get_data(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (spikes, calcium, adj_true) either from a saved dataset or
    by running a fresh simulation.

    - config.data_path is None  → simulate fresh, nothing saved
    - config.data_path is set   → SimulatedDataset.load_or_generate()
                                   (load if exists, simulate+save if not)
    """
    if config.data_path is not None:
        ds = SimulatedDataset.load_or_generate(config, config.data_path)
        return np.asarray(ds.spikes), np.asarray(ds.calcium), ds.adj_true

    # Fresh simulation — no saving
    net = BrunelNetwork(
        n_excitatory=config.n_excitatory,
        n_inhibitory=config.n_inhibitory,
        epsilon=config.epsilon,
        g=config.g,
        eta=config.eta,
        J_ex=config.J_ex,
        sim_time=config.sim_time,
        dt=config.dt,
        n_threads=config.n_threads,
    )
    net.build()
    net.run()
    spikes, adj_true = net.get_results()
    calcium = simulate_calcium(
        spikes,
        tau=config.tau,
        dt=config.dt,
        amplitude=config.amplitude,
        sigma_intra=config.sigma_intra,
        sigma_extra=config.sigma_extra,
        seed=config.seed,
    )
    return spikes, calcium, adj_true


def _call_solver(
    config: ExperimentConfig,
    zarr_path: str,
    lag_samples: int,
    on_epoch: Callable[[int, float], None] | None,
) -> np.ndarray:
    fn = getattr(_slv, _SOLVER_MAP[config.solver])
    common = dict(zarr_path=zarr_path, lag=lag_samples)

    if config.solver in ("ols", "sklearn_ols"):
        return fn(**common, chunk_size=config.chunk_size)

    if config.solver == "ridge":
        return fn(**common, lam=config.lam, chunk_size=config.chunk_size)

    if config.solver == "torch_normal_eq":
        return fn(**common, lam=config.lam, chunk_size=config.chunk_size,
                  device=config.device)

    if config.solver == "torch_gd":
        return fn(**common, chunk_size=config.chunk_size,
                  n_epochs=config.n_epochs, lr=config.lr,
                  device=config.device, on_epoch=on_epoch)

    if config.solver == "sklearn_lasso":
        return fn(**common, lam=config.lam, chunk_size=config.chunk_size)

    if config.solver == "fista":
        return fn(**common, lam_l1=config.lam, lam_l2=config.lam_l2,
                  chunk_size=config.chunk_size, n_iter=config.n_epochs * 50)

    return fn(
        **common,
        lam=config.lam,
        chunk_size=config.chunk_size,
        n_epochs=config.n_epochs,
        lr=config.lr,
        device=config.device,
        on_epoch=on_epoch,
    )


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def run_single(config: ExperimentConfig) -> ExperimentResult:
    """
    Execute the full pipeline for one configuration and persist the results.

    Parameters
    ----------
    config : ExperimentConfig

    Returns
    -------
    ExperimentResult
    """
    if config.solver not in _SOLVER_MAP:
        raise ValueError(
            f"Unknown solver '{config.solver}'. "
            f"Choose from: {list(_SOLVER_MAP)}"
        )

    t_start   = time.perf_counter()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    # ------------------------------------------------------------------ #
    # Derived sample-domain quantities from physical-unit config fields
    # ------------------------------------------------------------------ #
    lag_samples = max(1, round(config.lag_ms / config.dt))

    _w = max(5, round(config.smooth_window_ms / config.dt))
    smooth_win = _w if _w % 2 == 1 else _w + 1   # savgol requires odd window

    # ------------------------------------------------------------------ #
    # 1. Run directory
    # ------------------------------------------------------------------ #
    run_dir = make_run_dir(config)

    # ------------------------------------------------------------------ #
    # 2 & 3. Data load + preprocessing
    #
    # Two paths depending on whether a persistent dataset is on disk:
    #
    #   data_path set  →  streaming path: calcium stays in zarr, RAM is
    #                     O(N × chunk_size) regardless of signal length T.
    #                     Ideal for HPC runs with large N and long T.
    #
    #   data_path None →  in-memory path: simulate fresh, load everything
    #                     into RAM.  Fine for short exploratory runs.
    # ------------------------------------------------------------------ #
    if config.data_path is not None:
        # -------------------------------------------------------------- #
        # Streaming path
        # -------------------------------------------------------------- #
        ds = SimulatedDataset.load_or_generate(config, config.data_path)
        adj_true     = ds.adj_true
        np.fill_diagonal(adj_true, 0.0)
        calcium_zarr = ds.calcium           # zarr.Array — never fully loaded
        spikes       = None
        fluorescence = None

        # Estimate tau from the first 100k samples (fits comfortably in RAM)
        _N, _T = calcium_zarr.shape
        sub_T   = min(100_000, _T)
        cal_sub = np.asarray(calcium_zarr[:, :sub_T])
        tau_est = estimate_tau_robust(
            cal_sub,
            window_length=smooth_win,
            method=config.tau_method,
            dt=config.dt,
        )

        # Chunked feed reconstruction writes directly to zarr on disk
        feed_zarr_path = str(run_dir / "feed.zarr")
        stream_feed_to_zarr(
            calcium_zarr, feed_zarr_path, tau_est,
            window_length=smooth_win,
            chunk_size=config.chunk_size,
            dt=config.dt,
        )
    else:
        # -------------------------------------------------------------- #
        # In-memory path
        # -------------------------------------------------------------- #
        spikes, fluorescence, adj_true = _get_data(config)
        np.fill_diagonal(adj_true, 0.0)

        smooth, deriv = get_signal_derivative_pair(
            fluorescence, window_length=smooth_win, delta=config.dt
        )
        tau_est = estimate_tau_robust(
            fluorescence,
            window_length=smooth_win,
            method=config.tau_method,
            dt=config.dt,
        )
        feed = reconstruct_feed(smooth, deriv, tau_est)
        feed_zarr_path = save_feed_zarr(feed, run_dir)

    # ------------------------------------------------------------------ #
    # 5. Solve
    # ------------------------------------------------------------------ #
    loss_curve: list[float] = []
    adj_inferred = _call_solver(
        config, feed_zarr_path, lag_samples,
        on_epoch=lambda epoch, loss: loss_curve.append(loss),
    )
    np.fill_diagonal(adj_inferred, 0.0)

    # ------------------------------------------------------------------ #
    # 6. Evaluate — delegate entirely to metrics registry
    # ------------------------------------------------------------------ #
    all_metrics = compute_metrics(
        adj_inferred=adj_inferred,
        adj_true=adj_true,
        tau_est=np.atleast_1d(tau_est),
        tau_true=config.tau,
        loss_curve=loss_curve,
    )

    duration = time.perf_counter() - t_start

    # ------------------------------------------------------------------ #
    # 7. Build result
    # ------------------------------------------------------------------ #
    result = ExperimentResult(
        config_path       = str(run_dir / "config.json"),
        metrics           = all_metrics,
        loss_curve        = loss_curve,
        duration_seconds  = duration,
        timestamp         = timestamp,
        run_dir           = str(run_dir),
        # spikes / calcium are None in the streaming path (data lives in
        # the dataset zarr pointed to by config.data_path)
        spikes_path       = None if spikes is None else str(run_dir / "spikes.npy"),
        calcium_path      = None if fluorescence is None else str(run_dir / "calcium.npy"),
        feed_zarr_path    = feed_zarr_path,
        adj_true_path     = str(run_dir / "adj_true.npy"),
        adj_inferred_path = str(run_dir / "adj_inferred.npy"),
    )

    # ------------------------------------------------------------------ #
    # 8. Persist
    # ------------------------------------------------------------------ #
    save_result(result, config, spikes, fluorescence, adj_true, adj_inferred)

    # Append one row to the experiment ledger (Layer 1 of the record).
    # Non-fatal: a logging error must never lose a completed experiment.
    try:
        from .ledger import append_row
        append_row(result, config, Path(config.output_dir) / "ledger.csv")
    except Exception as e:                       # noqa: BLE001
        print(f"[ledger] warning: could not append run to ledger ({e})")

    corr = all_metrics.get("connectivity/pearson", float("nan"))
    auc  = all_metrics.get("connectivity/auc_roc", float("nan"))
    print(
        f"[{config.name}] done in {duration:.1f}s  "
        f"pearson={corr:.3f}  auc={auc:.3f}"
    )
    return result
