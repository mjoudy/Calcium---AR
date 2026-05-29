"""
Metrics registry for experiment evaluation.

Three registries cover the three domains where we measure quality:

    connectivity_metrics  — compare inferred A vs ground-truth adj
    tau_metrics           — compare estimated tau vs true tau
    solver_metrics        — analyse solver convergence curve

Registering a new metric
------------------------
Import the appropriate registry and decorate your function:

    from calcium_ar.experiments.metrics import connectivity_metrics

    @connectivity_metrics.register
    def my_metric(adj_inferred: np.ndarray, adj_true: np.ndarray) -> float:
        ...

It will then appear automatically in compute_all() output.

Computing all metrics
---------------------
    results = compute_all(
        adj_inferred=A,
        adj_true=adj,
        tau_est=tau_est,
        tau_true=100.0,
        loss_curve=[...],   # empty list for non-gradient solvers
    )
    # returns a flat dict: {'pearson': 0.71, 'auc_roc': 0.83, ...}
"""

from __future__ import annotations

from typing import Callable
import numpy as np


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

class MetricRegistry:
    """Holds a named collection of metric functions of identical signature."""

    def __init__(self, name: str):
        self.name = name
        self._metrics: dict[str, Callable] = {}

    def register(self, fn: Callable) -> Callable:
        """Decorator — adds fn to the registry under fn.__name__."""
        self._metrics[fn.__name__] = fn
        return fn

    def compute(self, *args, **kwargs) -> dict[str, float]:
        """Call every registered metric and return {name: value}."""
        out = {}
        for name, fn in self._metrics.items():
            try:
                out[name] = fn(*args, **kwargs)
            except Exception as e:
                out[name] = float("nan")
        return out

    def __repr__(self) -> str:
        return f"MetricRegistry('{self.name}', metrics={list(self._metrics)})"


# Three domain registries — import and decorate to add your own
connectivity_metrics = MetricRegistry("connectivity")
tau_metrics          = MetricRegistry("tau")
solver_metrics       = MetricRegistry("solver")


# --------------------------------------------------------------------------- #
# Built-in connectivity metrics
# (adj_inferred: N×N, adj_true: N×N)  →  float
# --------------------------------------------------------------------------- #

@connectivity_metrics.register
def pearson(adj_inferred: np.ndarray, adj_true: np.ndarray) -> float:
    """
    Pearson correlation between all off-diagonal entries.

    Convention note
    ---------------
    adj_true[s, t]   = synaptic weight from neuron s (source) to neuron t (target).
    adj_inferred[i, j] = AR coefficient: how strongly x_j(t-lag) predicts x_i(t),
                         i.e. the inferred weight FROM j TO i.

    Therefore adj_inferred[i, j]  ↔  adj_true[j, i]  =  adj_true.T[i, j].
    All connectivity metrics compare adj_inferred against adj_true.T.
    """
    N = adj_true.shape[0]
    mask = ~np.eye(N, dtype=bool)
    a = adj_inferred[mask].ravel()
    b = adj_true.T[mask].ravel()          # transpose: source→target convention
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@connectivity_metrics.register
def auc_roc(adj_inferred: np.ndarray, adj_true: np.ndarray) -> float:
    """
    Binary AUC-ROC: predicting whether a connection exists.
    score   = |adj_inferred|  (unsigned strength)
    label   = adj_true.T != 0   (see convention note in pearson())
    Diagonal (self-connections) excluded.
    """
    from sklearn.metrics import roc_auc_score
    N = adj_true.shape[0]
    mask = ~np.eye(N, dtype=bool)
    y_true  = (adj_true.T[mask] != 0).astype(int)  # transpose: source→target convention
    y_score = np.abs(adj_inferred[mask])
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


@connectivity_metrics.register
def mse(adj_inferred: np.ndarray, adj_true: np.ndarray) -> float:
    """Mean squared error between inferred and true connectivity (adj_true.T convention)."""
    return float(np.mean((adj_inferred - adj_true.T) ** 2))


@connectivity_metrics.register
def sparsity_ratio(adj_inferred: np.ndarray, adj_true: np.ndarray) -> float:
    """
    Ratio of inferred sparsity to true sparsity (adj_true.T convention).
    A value close to 1 means the solver recovered the right density.
    Inferred weight is considered non-zero if |w| > 1% of max |w|.
    """
    N = adj_true.shape[0]
    mask = ~np.eye(N, dtype=bool)
    true_nnz = (adj_true.T[mask] != 0).sum()
    if true_nnz == 0:
        return float("nan")
    threshold = 0.01 * np.abs(adj_inferred[mask]).max()
    inferred_nnz = (np.abs(adj_inferred[mask]) > threshold).sum()
    return float(inferred_nnz / true_nnz)


# --------------------------------------------------------------------------- #
# Built-in tau metrics
# (tau_est: float | np.ndarray, tau_true: float)  →  float
# --------------------------------------------------------------------------- #

@tau_metrics.register
def tau_mae(tau_est: np.ndarray | float, tau_true: float) -> float:
    """Mean absolute error of tau estimation across neurons."""
    return float(np.mean(np.abs(np.asarray(tau_est) - tau_true)))


@tau_metrics.register
def tau_mse(tau_est: np.ndarray | float, tau_true: float) -> float:
    """Mean squared error of tau estimation."""
    return float(np.mean((np.asarray(tau_est) - tau_true) ** 2))


@tau_metrics.register
def tau_relative_error(tau_est: np.ndarray | float, tau_true: float) -> float:
    """Mean relative error: mean(|tau_est - tau_true| / tau_true)."""
    if tau_true == 0:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(tau_est) - tau_true) / tau_true))


# --------------------------------------------------------------------------- #
# Built-in solver metrics
# (loss_curve: list[float])  →  float
# --------------------------------------------------------------------------- #

@solver_metrics.register
def final_loss(loss_curve: list[float]) -> float:
    """Loss value at the last epoch."""
    return float(loss_curve[-1]) if loss_curve else float("nan")


@solver_metrics.register
def convergence_ratio(loss_curve: list[float]) -> float:
    """
    Ratio of final loss to initial loss.
    < 1 means the solver improved; close to 0 means strong convergence.
    """
    if not loss_curve or loss_curve[0] == 0:
        return float("nan")
    return float(loss_curve[-1] / loss_curve[0])


@solver_metrics.register
def loss_auc(loss_curve: list[float]) -> float:
    """Area under the loss curve (trapezoidal) — lower is better."""
    if not loss_curve:
        return float("nan")
    return float(np.trapz(loss_curve))


# --------------------------------------------------------------------------- #
# Aggregate entry point
# --------------------------------------------------------------------------- #

def compute_all(
    adj_inferred: np.ndarray,
    adj_true: np.ndarray,
    tau_est: np.ndarray | float | None = None,
    tau_true: float | None = None,
    loss_curve: list[float] | None = None,
) -> dict[str, float]:
    """
    Run all registered metrics and return a single flat dict.

    Keys follow the pattern  <registry_name>/<metric_name>
    so they stay unambiguous even if two registries share a name.

    Parameters
    ----------
    adj_inferred : (N, N) inferred connectivity matrix.
    adj_true     : (N, N) ground-truth connectivity matrix.
    tau_est      : per-neuron or scalar tau estimate (optional).
    tau_true     : scalar true tau used in simulation (optional).
    loss_curve   : list of per-epoch losses from gradient solvers (optional).

    Returns
    -------
    dict[str, float]  — flat mapping of metric name → value.
    """
    results: dict[str, float] = {}

    for name, val in connectivity_metrics.compute(adj_inferred, adj_true).items():
        results[f"connectivity/{name}"] = val

    if tau_est is not None and tau_true is not None:
        for name, val in tau_metrics.compute(tau_est, tau_true).items():
            results[f"tau/{name}"] = val

    if loss_curve is not None:
        for name, val in solver_metrics.compute(loss_curve).items():
            results[f"solver/{name}"] = val

    return results
