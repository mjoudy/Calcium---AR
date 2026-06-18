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
def spearman(adj_inferred: np.ndarray, adj_true: np.ndarray) -> float:
    """
    Spearman rank-order correlation between off-diagonal entries (adj_true.T convention).

    More appropriate than Pearson here because adj_true has only 3 discrete values
    (0, +J_ex, -g·J_ex) — far from Gaussian — and the mapping G → A = e^{GΔ} is
    monotonic but not linear, so rank-based agreement is the right measure.
    """
    from scipy.stats import spearmanr
    N = adj_true.shape[0]
    mask = ~np.eye(N, dtype=bool)
    a = adj_inferred[mask].ravel()
    b = adj_true.T[mask].ravel()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    corr, _ = spearmanr(a, b)
    return float(corr)


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
    """Mean squared error between off-diagonal entries (adj_true.T convention)."""
    N = adj_true.shape[0]
    mask = ~np.eye(N, dtype=bool)
    return float(np.mean((adj_inferred[mask] - adj_true.T[mask]) ** 2))


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
# Thresholding helpers (private — not registered)                             #
# --------------------------------------------------------------------------- #
# An inferred weight is declared "connected" if |w| > _CONN_THR_FRAC * max|w|.
# For solvers that produce exact zeros (FISTA, Lasso) this threshold is moot;
# for dense solvers (OLS, Ridge) it defines the decision boundary.
_CONN_THR_FRAC = 0.01


def _binary_labels(
    adj_inferred: np.ndarray, adj_true: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Off-diagonal binary labels: 1 = connected, 0 = unconnected."""
    N = adj_true.shape[0]
    mask  = ~np.eye(N, dtype=bool)
    y_true = (adj_true.T[mask] != 0).astype(int)
    thr    = _CONN_THR_FRAC * np.abs(adj_inferred[mask]).max()
    y_pred = (np.abs(adj_inferred[mask]) > thr).astype(int)
    return y_true, y_pred


def _ternary_labels(
    adj_inferred: np.ndarray, adj_true: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Off-diagonal ternary labels: +1 = excitatory, -1 = inhibitory, 0 = none."""
    N = adj_true.shape[0]
    mask  = ~np.eye(N, dtype=bool)
    y_true = np.sign(adj_true.T[mask]).astype(int)
    a_off  = adj_inferred[mask]
    thr    = _CONN_THR_FRAC * np.abs(a_off).max()
    y_pred = np.where(np.abs(a_off) > thr, np.sign(a_off).astype(int), 0)
    return y_true, y_pred


# --------------------------------------------------------------------------- #
# Binary confusion-matrix metrics                                              #
# (connected vs unconnected, off-diagonal)                                    #
# --------------------------------------------------------------------------- #

@connectivity_metrics.register
def precision(adj_inferred: np.ndarray, adj_true: np.ndarray) -> float:
    """TP / (TP + FP) — fraction of predicted connections that are real."""
    y_true, y_pred = _binary_labels(adj_inferred, adj_true)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")


@connectivity_metrics.register
def recall(adj_inferred: np.ndarray, adj_true: np.ndarray) -> float:
    """TP / (TP + FN) — fraction of true connections that were found."""
    y_true, y_pred = _binary_labels(adj_inferred, adj_true)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")


@connectivity_metrics.register
def f1(adj_inferred: np.ndarray, adj_true: np.ndarray) -> float:
    """Harmonic mean of precision and recall."""
    y_true, y_pred = _binary_labels(adj_inferred, adj_true)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(2.0 * p * r / (p + r)) if (p + r) > 0 else float("nan")


@connectivity_metrics.register
def specificity(adj_inferred: np.ndarray, adj_true: np.ndarray) -> float:
    """TN / (TN + FP) — fraction of true non-connections correctly rejected."""
    y_true, y_pred = _binary_labels(adj_inferred, adj_true)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Arxiv custom metrics                                                         #
# --------------------------------------------------------------------------- #

@connectivity_metrics.register
def performance_score(adj_inferred: np.ndarray, adj_true: np.ndarray) -> float:
    """
    Mean per-class recall across excitatory / inhibitory / unconnected classes.

    Original definition from the _arxiv pipeline (conn_inf_funcs.py):
        score = sum_i [ CM[i,i] / sum_j CM[i,j] ]  (sum of per-class recalls)
    Normalised here by the number of classes present so the result is in [0, 1].
    A score of 1 means every class was perfectly recovered.
    """
    y_true, y_pred = _ternary_labels(adj_inferred, adj_true)
    classes = np.unique(y_true)
    if len(classes) < 2:
        return float("nan")
    score = 0.0
    for cls in classes:
        mask_cls = y_true == cls
        n_cls = mask_cls.sum()
        if n_cls == 0:
            continue
        score += float((y_pred[mask_cls] == cls).sum()) / n_cls
    return score / len(classes)


@connectivity_metrics.register
def exc_unc_overlap(adj_inferred: np.ndarray, adj_true: np.ndarray) -> float:
    """
    Size of the overlap interval between the excitatory and unconnected
    inferred-weight distributions (original: overlap_interval in _arxiv).

    Smaller = more separable = better.  0 means the two distributions do not
    overlap at all.  A positive value is the width of the overlap band.
    """
    N = adj_true.shape[0]
    mask   = ~np.eye(N, dtype=bool)
    g_flat = adj_true.T[mask]
    a_flat = adj_inferred[mask]
    A_exc  = a_flat[g_flat > 0]
    A_unc  = a_flat[g_flat == 0]
    if len(A_exc) == 0 or len(A_unc) == 0:
        return float("nan")
    overlap_lo = max(float(A_exc.min()), float(A_unc.min()))
    overlap_hi = min(float(A_exc.max()), float(A_unc.max()))
    return max(0.0, overlap_hi - overlap_lo)


@connectivity_metrics.register
def inh_unc_overlap(adj_inferred: np.ndarray, adj_true: np.ndarray) -> float:
    """
    Size of the overlap interval between the inhibitory and unconnected
    inferred-weight distributions (original: overlap_interval in _arxiv).

    Smaller = more separable = better.
    """
    N = adj_true.shape[0]
    mask   = ~np.eye(N, dtype=bool)
    g_flat = adj_true.T[mask]
    a_flat = adj_inferred[mask]
    A_inh  = a_flat[g_flat < 0]
    A_unc  = a_flat[g_flat == 0]
    if len(A_inh) == 0 or len(A_unc) == 0:
        return float("nan")
    overlap_lo = max(float(A_inh.min()), float(A_unc.min()))
    overlap_hi = min(float(A_inh.max()), float(A_unc.max()))
    return max(0.0, overlap_hi - overlap_lo)


# --------------------------------------------------------------------------- #
# Dale's Law metrics                                                           #
# --------------------------------------------------------------------------- #
# Both metrics treat column j of adj_inferred as the outgoing weights of
# neuron j  (A[i,j] = inferred weight j → i), which is consistent with the
# adj_inferred[i,j] ↔ adj_true.T[i,j] convention used throughout.
#
# True neuron type is read from row j of adj_true (adj_true[j,:] = outgoing
# weights of neuron j in the source→target convention).  Neurons whose entire
# outgoing row is zero are skipped in Option 2 (no ground-truth label).

@connectivity_metrics.register
def degree_of_daleianity(adj_inferred: np.ndarray, adj_true: np.ndarray) -> float:
    """
    Degree of Daleianity  (unsupervised, no ground truth needed).

    For each neuron j, apply the coefficient D to its outgoing weight column:

        D(x) = |∑ x_i| / ∑ |x_i|

    where x is the off-diagonal column j of adj_inferred.  Then average D
    across all neurons whose column is not identically zero.

    Properties:
      D(x) ∈ [0, 1]
      D(x) = 1  ⟺  all x_i share the same sign  (perfect Dale compliance)
      D(x) = 0  ⟺  positive and negative entries perfectly cancel
      D(x) is magnitude-weighted: large connections dominate, small noise
            entries of random sign contribute little — no threshold needed.
    """
    N = adj_inferred.shape[0]
    scores = []
    for j in range(N):
        col    = adj_inferred[:, j].copy()
        col[j] = 0.0                        # exclude self-connection
        denom  = np.abs(col).sum()
        if denom == 0.0:
            continue                         # silent neuron — skip
        scores.append(abs(col.sum()) / denom)
    return float(np.mean(scores)) if scores else float("nan")


@connectivity_metrics.register
def dale_type_accuracy(adj_inferred: np.ndarray, adj_true: np.ndarray) -> float:
    """
    Neuron-type recovery accuracy  (Option 2 — supervised).

    For each neuron j, derive its true type from the sign of the sum of row j
    in adj_true (positive sum → excitatory, negative sum → inhibitory).
    Assign the inferred type by majority vote of column j's non-zero weights.
    Return the fraction of neurons whose inferred type matches the true type.

    Neurons with no non-zero outgoing connections in adj_true (genuinely
    ambiguous ground truth) are excluded from the denominator.
    Range [0, 1] — 1.0 means all neuron types correctly identified.
    """
    N   = adj_inferred.shape[0]
    thr = _CONN_THR_FRAC * np.abs(adj_inferred[~np.eye(N, dtype=bool)]).max()
    correct = 0
    total   = 0
    for j in range(N):
        # True type: sign of the sum of outgoing row in adj_true
        row_j      = adj_true[j].copy();  row_j[j] = 0.0
        true_sum   = row_j.sum()
        if true_sum == 0:
            continue                     # no ground-truth label for this neuron
        true_type  = 1 if true_sum > 0 else -1

        # Inferred type: majority vote among non-zero outgoing column entries
        col_j      = adj_inferred[:, j].copy();  col_j[j] = 0.0
        n_pos      = int((col_j >  thr).sum())
        n_neg      = int((col_j < -thr).sum())
        if n_pos + n_neg == 0:
            continue                     # solver zeroed out everything — skip
        inferred_type = 1 if n_pos >= n_neg else -1

        correct += int(true_type == inferred_type)
        total   += 1
    return float(correct / total) if total > 0 else float("nan")


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
