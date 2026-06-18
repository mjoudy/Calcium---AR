"""
Feed (spike proxy) reconstruction from calcium signals.

Given the calcium ODE:
    dC/dt = -C/tau + S(t)

rearranging gives:
    S(t) ≈ dC/dt + (1/tau) * C

i.e. the estimated feed equals the derivative plus C scaled by the decay rate.
"""

from __future__ import annotations

import numpy as np


def reconstruct_feed(
    signal: np.ndarray,
    deriv: np.ndarray,
    tau: float | np.ndarray,
) -> np.ndarray:
    """
    Reconstruct the spike-drive (feed) from smoothed calcium and its derivative.

    Parameters
    ----------
    signal : (T,) or (N, T) — smoothed calcium signal.
    deriv  : same shape — first derivative of signal.
    tau    : float or (N,) array — decay time constant(s).
              If scalar, the same tau is used for all neurons.

    Returns
    -------
    feed : same shape as signal.
        Estimated input drive; positive peaks correspond to spike events.
    """
    signal = np.asarray(signal, dtype=float)
    deriv = np.asarray(deriv, dtype=float)
    tau = np.asarray(tau, dtype=float)

    if signal.ndim == 2 and tau.ndim == 1:
        # Broadcast tau (N,) over time axis: shape (N, 1)
        tau = tau[:, np.newaxis]

    return deriv + signal / tau


def stream_feed_to_zarr(
    calcium_zarr,
    out_path: str,
    tau_est: np.ndarray | float,
    window_length: int = 31,
    chunk_size: int = 10_000,
    dt: float = 1.0,
) -> str:
    """
    Chunked streaming feed reconstruction — O(N × chunk_size) RAM.

    Reads the calcium zarr array in overlapping windows, applies
    Savitzky-Golay smoothing and derivative, reconstructs
    feed = deriv + signal/tau, and writes the result to a new zarr
    at out_path.

    Boundary overlap
    ----------------
    Each chunk is extended by half_win = window_length // 2 samples on
    each side before the SG filter is applied.  Only the central
    (non-overlapping) samples are written to the output, so SG edge
    artefacts are discarded for every interior sample.  The first and
    last half_win samples of the full signal are handled with scipy's
    default 'interp' boundary mode — identical to the in-memory path.

    Parameters
    ----------
    calcium_zarr : zarr.Array of shape (N, T)
    out_path     : destination zarr path (created/overwritten)
    tau_est      : (N,) or scalar — decay time constants in sample units
    window_length: SG window length (same value used for tau estimation)
    chunk_size   : core time-steps per iteration (set to available RAM / (N×8 bytes))

    Returns
    -------
    out_path (str)
    """
    import zarr
    import scipy.signal as sig

    N, T = calcium_zarr.shape
    half_win = window_length // 2

    tau_col = np.asarray(tau_est, dtype=float)
    if tau_col.ndim == 1:
        tau_col = tau_col[:, np.newaxis]   # (N, 1) — broadcast over time axis

    chunk_t = min(chunk_size, T)
    z_out = zarr.open_array(
        out_path, mode="w", shape=(N, T), dtype=np.float64,
        chunks=(N, chunk_t),
    )

    t0 = 0
    while t0 < T:
        t1 = min(t0 + chunk_size, T)

        # Extend chunk boundaries to absorb SG filter edge artefacts
        ext_start = max(0, t0 - half_win)
        ext_end   = min(T, t1 + half_win)

        block = np.asarray(calcium_zarr[:, ext_start:ext_end], dtype=float)

        smooth = sig.savgol_filter(
            block, window_length=window_length, polyorder=3,
            deriv=0, delta=dt, axis=-1,
        )
        deriv = sig.savgol_filter(
            block, window_length=window_length, polyorder=3,
            deriv=1, delta=dt, axis=-1,
        )

        # Slice the core (non-overlapping) region
        core_start = t0 - ext_start
        core_end   = core_start + (t1 - t0)

        z_out[:, t0:t1] = deriv[:, core_start:core_end] + smooth[:, core_start:core_end] / tau_col

        t0 = t1

    return out_path
