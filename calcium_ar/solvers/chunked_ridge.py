"""
Chunked Ridge (Tikhonov) regression solver with mean-centring.

Solves  X_t = A @ X_{t-1}  with L2 regularisation on *mean-centred* data:

    A = C_yx_c @ inv(C_xx_c + λI)

Mean-centring is applied using the same one-pass correction as chunked_ols:

    C_xx_c = C_xx_raw - (1/n) * s_prev ⊗ s_prev
    C_yx_c = C_yx_raw - (1/n) * s_now  ⊗ s_prev

See chunked_ols.py for a full explanation of why centring is necessary.

Adding λI to C_xx_c before inversion:
  - prevents division by near-zero eigenvalues (ill-conditioning)
  - shrinks all weights of A toward zero proportionally
  - does NOT produce exact zeros — A remains dense

λ = 0 recovers plain OLS (chunked_ols.solve).
λ → ∞ forces A → 0.

RAM and time cost are identical to chunked_ols — the only extra cost is
adding λ to the diagonal of the (N×N) C_xx_c before np.linalg.solve.
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
    lam        : L2 regularisation strength λ  (0 = plain OLS)
    chunk_size : number of time steps loaded per iteration

    Returns
    -------
    A : np.ndarray, shape (N, N)
        Estimated connectivity matrix (dense, L2-regularised).
    """
    signals = zarr.open(zarr_path, mode="r")
    N, T = signals.shape

    # ------------------------------------------------------------------ #
    # Accumulators                                                         #
    # ------------------------------------------------------------------ #
    C_xx_raw = np.zeros((N, N))
    C_yx_raw = np.zeros((N, N))
    s_prev   = np.zeros(N)
    s_now    = np.zeros(N)
    n        = 0

    n_chunks = (T - lag) // chunk_size

    # ------------------------------------------------------------------ #
    # Single pass: accumulate raw cross-products and column sums          #
    # ------------------------------------------------------------------ #
    for k in range(n_chunks):
        t0 = k * chunk_size
        t1 = t0 + chunk_size

        x_prev = np.asarray(signals[:, t0       : t1      ])
        x_now  = np.asarray(signals[:, t0 + lag : t1 + lag])

        C_xx_raw += x_prev @ x_prev.T
        C_yx_raw += x_now  @ x_prev.T

        s_prev   += x_prev.sum(axis=1)
        s_now    += x_now .sum(axis=1)
        n        += x_prev.shape[1]

    # ------------------------------------------------------------------ #
    # Mean-centring correction (equivalent to sklearn fit_intercept=True) #
    # ------------------------------------------------------------------ #
    mu_prev = s_prev / n
    mu_now  = s_now  / n

    C_xx_c = C_xx_raw - n * np.outer(mu_prev, mu_prev)
    C_yx_c = C_yx_raw - n * np.outer(mu_now,  mu_prev)

    # ------------------------------------------------------------------ #
    # Solve the regularised normal equations on the centred matrices      #
    # ------------------------------------------------------------------ #
    C_xx_reg = C_xx_c + lam * np.eye(N)
    A = np.linalg.solve(C_xx_reg.T, C_yx_c.T).T
    return A
