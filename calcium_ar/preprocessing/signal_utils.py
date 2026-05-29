"""
Signal smoothing, derivative computation, and spike-window removal.
"""

import numpy as np
import scipy.signal as sig


def smooth_signal(
    signal: np.ndarray,
    window_length: int = 5,
    polyorder: int = 3,
    deriv: int = 0,
    delta: float = 1.0,
) -> np.ndarray:
    """Savitzky-Golay smoothing / derivative. Works on 1D (T,) or 2D (N, T)."""
    return sig.savgol_filter(
        np.asarray(signal),
        window_length=window_length,
        polyorder=polyorder,
        deriv=deriv,
        delta=delta,
        axis=-1,
    )


def get_signal_derivative_pair(
    signal: np.ndarray,
    window_length: int = 5,
    delta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (smoothed_signal, first_derivative) computed in one pass."""
    signal = np.asarray(signal)
    smooth = smooth_signal(signal, window_length=window_length, deriv=0, delta=delta)
    deriv = smooth_signal(signal, window_length=window_length, deriv=1, delta=delta)
    return smooth, deriv


def cut_spikes(
    spikes: np.ndarray,
    signal: np.ndarray,
    deriv: np.ndarray | None = None,
    window_length: int = 5,
) -> tuple | list:
    """
    Remove spike-window samples from signal (and optionally derivative).

    Because spike counts differ per neuron the output is ragged: a list of
    1-D arrays for 2-D input, or a single 1-D array for 1-D input.

    Parameters
    ----------
    spikes : (N, T) or (T,) — binary (0/1) or indices of spike times.
    signal : same shape as spikes.
    deriv  : same shape as spikes (optional).
    window_length : half-window around each spike to remove (samples).

    Returns
    -------
    cut_signal [, cut_deriv]  — list of 1-D arrays (2-D input) or single array.
    """
    spikes = np.asarray(spikes)
    signal = np.asarray(signal)
    input_1d = spikes.ndim == 1

    spikes_2d = np.atleast_2d(spikes)
    signal_2d = np.atleast_2d(signal)
    if deriv is not None:
        deriv = np.asarray(deriv)
        deriv_2d = np.atleast_2d(deriv)

    N, T = spikes_2d.shape
    cut_signals, cut_derivs = [], []

    for i in range(N):
        row = spikes_2d[i]
        spike_idx = np.where(row)[0] if np.isin(row, [0, 1]).all() else row.astype(int)

        if len(spike_idx) > 0:
            windows = spike_idx[:, None] + np.arange(-window_length, window_length)
            remove = np.unique(np.clip(windows.ravel(), 0, T - 1))
            cut_s = np.delete(signal_2d[i], remove)
            cut_d = np.delete(deriv_2d[i], remove) if deriv is not None else None
        else:
            cut_s = signal_2d[i].copy()
            cut_d = deriv_2d[i].copy() if deriv is not None else None

        cut_signals.append(cut_s)
        if deriv is not None:
            cut_derivs.append(cut_d)

    if input_1d:
        if deriv is not None:
            return cut_signals[0], cut_derivs[0]
        return cut_signals[0]

    if deriv is not None:
        return cut_signals, cut_derivs
    return cut_signals


def calculate_cv(spikes: np.ndarray) -> float | np.ndarray:
    """Coefficient of variation of inter-spike intervals."""
    spikes = np.asarray(spikes)
    input_1d = spikes.ndim == 1
    spikes_2d = np.atleast_2d(spikes)
    cvs = []
    for row in spikes_2d:
        times = np.where(row)[0] if np.isin(row, [0, 1]).all() else row.astype(int)
        if len(times) < 2:
            cvs.append(np.nan)
            continue
        isis = np.diff(times)
        mu = isis.mean()
        cvs.append(0.0 if mu == 0 else isis.std() / mu)
    out = np.array(cvs)
    return out[0] if input_1d else out


def calculate_cv2(spikes: np.ndarray) -> float | np.ndarray:
    """Local coefficient of variation (CV2) of inter-spike intervals."""
    spikes = np.asarray(spikes)
    input_1d = spikes.ndim == 1
    spikes_2d = np.atleast_2d(spikes)
    cv2s = []
    for row in spikes_2d:
        times = np.where(row)[0] if np.isin(row, [0, 1]).all() else row.astype(int)
        if len(times) < 3:
            cv2s.append(np.nan)
            continue
        isis = np.diff(times)
        num = 2 * np.abs(np.diff(isis))
        denom = isis[:-1] + isis[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            vals = np.where(denom == 0, 0.0, num / denom)
        cv2s.append(float(np.nanmean(vals)))
    out = np.array(cv2s)
    return out[0] if input_1d else out
