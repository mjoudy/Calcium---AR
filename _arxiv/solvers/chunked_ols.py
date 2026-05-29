"""
Chunked Ordinary Least Squares (OLS) solver.

Solves  X_t = A @ X_{t-1}  via the normal equations:

    A = C_yx @ inv(C_xx)

where
    C_xx = X_{t-1} X_{t-1}^T      (N x N)
    C_yx = X_t    X_{t-1}^T       (N x N)

Both matrices are accumulated one chunk at a time from Zarr, so the full
(T x N) design matrix is never loaded into RAM.

RAM cost : ~2 * N^2 * 8 bytes  (two N x N accumulators) + one chunk at a time
Time cost: O(T * N^2)  — identical to sklearn OLS, faster in practice because
           numpy BLAS uses all CPU cores on every chunk multiply.
"""

import numpy as np
import zarr


def solve(zarr_path: str, lag: int = 10, chunk_size: int = 10_000) -> np.ndarray:
    """
    Parameters
    ----------
    zarr_path  : path to preprocessed signals array, shape (N, T)
    lag        : AR lag — pairs x(t) with x(t - lag)
    chunk_size : number of time steps loaded per iteration (~100 MB for N=1250)

    Returns
    -------
    A : np.ndarray, shape (N, N)
        Estimated connectivity matrix (dense, no regularization).
    """
    signals = zarr.open(zarr_path, mode="r")
    N, T = signals.shape

    C_xx = np.zeros((N, N))
    C_yx = np.zeros((N, N))

    n_chunks = (T - lag) // chunk_size

    for k in range(n_chunks):
        t0 = k * chunk_size
        t1 = t0 + chunk_size

        x_prev = np.asarray(signals[:, t0       : t1      ])   # (N, chunk_size)
        x_now  = np.asarray(signals[:, t0 + lag : t1 + lag])   # (N, chunk_size)

        C_xx += x_prev @ x_prev.T   # (N, N) — BLAS uses all CPU cores
        C_yx += x_now  @ x_prev.T   # (N, N)

    # Solve C_xx.T A.T = C_yx.T  (numerically preferred over explicit inv)
    A = np.linalg.solve(C_xx.T, C_yx.T).T
    return A
