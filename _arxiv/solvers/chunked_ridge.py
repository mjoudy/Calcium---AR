"""
Chunked Ridge (Tikhonov) regression solver.

Solves  X_t = A @ X_{t-1}  with L2 regularization:

    A = C_yx @ inv(C_xx + λI)

Adding λI to C_xx before inversion:
  - prevents division by near-zero eigenvalues (ill-conditioning)
  - shrinks all weights of A toward zero proportionally
  - does NOT produce exact zeros — A remains dense

λ = 0 recovers plain OLS (chunked_ols.solve).
λ → ∞ forces A → 0.

RAM and time cost are identical to chunked_ols — the only difference is
adding λ to the diagonal of the (N x N) C_xx before np.linalg.solve.
"""

import numpy as np
import zarr


def solve(zarr_path: str, lag: int = 10, lam: float = 1.0,
          chunk_size: int = 10_000) -> np.ndarray:
    """
    Parameters
    ----------
    zarr_path  : path to preprocessed signals array, shape (N, T)
    lag        : AR lag — pairs x(t) with x(t - lag)
    lam        : L2 regularization strength λ  (0 = plain OLS)
    chunk_size : number of time steps loaded per iteration

    Returns
    -------
    A : np.ndarray, shape (N, N)
        Estimated connectivity matrix (dense, L2-regularized).
    """
    signals = zarr.open(zarr_path, mode="r")
    N, T = signals.shape

    C_xx = np.zeros((N, N))
    C_yx = np.zeros((N, N))

    n_chunks = (T - lag) // chunk_size

    for k in range(n_chunks):
        t0 = k * chunk_size
        t1 = t0 + chunk_size

        x_prev = np.asarray(signals[:, t0       : t1      ])
        x_now  = np.asarray(signals[:, t0 + lag : t1 + lag])

        C_xx += x_prev @ x_prev.T
        C_yx += x_now  @ x_prev.T

    C_xx_reg = C_xx + lam * np.eye(N)
    A = np.linalg.solve(C_xx_reg.T, C_yx.T).T
    return A
