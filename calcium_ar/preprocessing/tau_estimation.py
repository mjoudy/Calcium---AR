"""
Calcium decay time-constant (tau) estimation via phase-space regression.

The calcium ODE is  dC/dt = -C/tau + S(t), so in phase space
(signal vs. derivative) the bulk of the trajectory lies on a line with
slope  m = -1/tau.  Spike events create looping outliers; robust methods
isolate the bulk slope and convert it to tau.
"""

import numpy as np
from .signal_utils import get_signal_derivative_pair
from .outliers import (
    fit_linear,
    remove_outliers_ransac,
    remove_outliers_dbscan,
    remove_outliers_iqr,
    remove_outliers_zscore,
)

_METHODS = {
    "ols": fit_linear,
    "ransac": remove_outliers_ransac,
    "dbscan": remove_outliers_dbscan,
    "iqr": remove_outliers_iqr,
    "zscore": remove_outliers_zscore,
}


def _slope_to_tau(slopes):
    """Convert phase-space slope(s) m = -1/tau → tau = -1/m."""
    slopes = np.asarray(slopes, dtype=float)
    scalar = slopes.ndim == 0
    slopes = np.atleast_1d(slopes)
    taus = np.zeros_like(slopes)
    nz = np.abs(slopes) > 1e-9
    taus[nz] = -1.0 / slopes[nz]
    return float(taus[0]) if scalar else taus


def estimate_time_constant(
    cut_signal: np.ndarray,
    cut_deriv: np.ndarray,
) -> float | np.ndarray:
    """
    Estimate tau from already-spike-cut signal and derivative pairs.

    Parameters
    ----------
    cut_signal : 1-D array or list of 1-D arrays (spike windows removed).
    cut_deriv  : matching derivative array(s).

    Returns
    -------
    tau : float or np.ndarray
    """
    slopes = fit_linear(cut_signal, cut_deriv)
    return _slope_to_tau(slopes)


def estimate_tau_robust(
    signal: np.ndarray,
    window_length: int = 5,
    method: str = "ransac",
    dt: float = 1.0,
    do_plot: bool = False,
) -> float | np.ndarray:
    """
    Estimate tau directly from raw calcium signal without explicit spike cutting.

    Robust regression on the phase space (signal, derivative) identifies the
    exponential-decay bulk while ignoring spike-trajectory outliers.

    Parameters
    ----------
    signal : (T,) or (N, T) calcium fluorescence.
    window_length : Savitzky-Golay window for derivative estimation (samples).
    method : one of 'ols', 'ransac' (default), 'dbscan', 'iqr', 'zscore'.
    dt : time step in ms. Passed as delta to savgol so the derivative is in
         signal/ms and the returned tau is in ms.
    do_plot : show phase-space plot for the first neuron.

    Returns
    -------
    tau : float (1-D input) or np.ndarray shape (N,) — in the same units as dt.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {list(_METHODS)}; got '{method}'")

    smooth, deriv = get_signal_derivative_pair(signal, window_length=window_length,
                                               delta=dt)
    slopes = _METHODS[method](smooth, deriv, do_plot=do_plot)
    return _slope_to_tau(slopes)
