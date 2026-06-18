"""
OLS via sklearn.LinearRegression (fit_intercept=True).

Loads the full (T-lag) × N design matrix into RAM, then solves all N
target regressions in a single LAPACK call.

RAM cost : O(T × N) — two matrices of shape (T-lag, N)
           ≈ 800 MB at T=500k, N=100 (float64)
Time cost: O(T × N²) QR factorisation — similar to chunked_ols but
           less cache-friendly for large T

This matches what the arxiv HPC scripts used
(LinearRegression(n_jobs=-1).fit(X_prev, X_now)).
fit_intercept=True is mathematically equivalent to the mean-centring
correction applied in chunked_ols, so both solvers should return the
same A up to floating-point rounding.
"""

import numpy as np
import zarr
from sklearn.linear_model import LinearRegression


def solve(
    zarr_path: str,
    lag: int = 10,
    chunk_size: int = 10_000,   # unused — kept for uniform API
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Parameters
    ----------
    zarr_path  : path to preprocessed signals array, shape (N, T)
    lag        : AR lag — pairs x(t) with x(t - lag)
    chunk_size : ignored (full matrix is loaded); kept for API consistency
    n_jobs     : passed to LinearRegression (multi-output parallelism)

    Returns
    -------
    A : np.ndarray, shape (N, N)
        Estimated connectivity matrix.
        A[i, j] = inferred weight from neuron j to neuron i.
    """
    signals = zarr.open(zarr_path, mode="r")
    N, T = signals.shape

    data   = np.asarray(signals, dtype=float)   # (N, T)
    x_prev = data[:, :T - lag].T                # (T-lag, N) — features
    x_now  = data[:, lag:].T                    # (T-lag, N) — targets

    reg = LinearRegression(fit_intercept=True, n_jobs=n_jobs)
    reg.fit(x_prev, x_now)
    return reg.coef_   # (N, N): coef_[i, j] = weight from feature j to target i
