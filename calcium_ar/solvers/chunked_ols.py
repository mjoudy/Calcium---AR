"""
Chunked Ordinary Least Squares (OLS) solver with mean-centring.

Solves  X_t = A @ X_{t-1}  via the normal equations on *mean-centred* data:

    A = C_yx_c @ inv(C_xx_c)

where
    C_xx_c = (X_{t-1} - μ)(X_{t-1} - μ)^T    centred lag-1 auto-covariance  (N×N)
    C_yx_c = (X_t    - μ)(X_{t-1} - μ)^T      centred cross-covariance       (N×N)

WHY MEAN-CENTRING MATTERS
--------------------------
Without centring, the raw cross-product matrices contain a large rank-1 bias:

    C_xx_raw = C_xx_c  +  n · μ μ^T
    C_yx_raw = C_yx_c  +  n · μ μ^T

For calcium-derived feed signals the per-neuron mean μ is significantly
positive (the C/τ term in  S ≈ dC/dt + C/τ  contributes a positive offset).
The outer-product term n·μμ^T dominates the raw matrices and swamps the true
covariance structure, reducing the Pearson correlation of the estimated
connectivity matrix to near zero.

Centring is mathematically equivalent to including a free intercept in the
regression — exactly what sklearn.LinearRegression does by default
(fit_intercept=True).  The archive HPC scripts that achieved good results all
used sklearn with the default intercept; this solver replicates that behaviour
without loading the full (T×N) design matrix into memory.

ONE-PASS CORRECTION
--------------------
Centring is applied after accumulation using the algebraic identity:

    C_xx_c = C_xx_raw - (1/n) * s_prev ⊗ s_prev
    C_yx_c = C_yx_raw - (1/n) * s_now  ⊗ s_prev

where  s_prev = Σ_t x_{t-1}  and  s_now = Σ_t x_t  are the column sums
and  ⊗  denotes the outer product.  This requires only a single Zarr pass.

RAM cost : ~2·N²·8 bytes  (two N×N accumulators) + two N-vectors + one chunk
Time cost: O(T·N²) — single Zarr pass, same complexity as sklearn OLS
"""

import numpy as np
import zarr


def solve(zarr_path: str, lag: int = 10, chunk_size: int = 10_000) -> np.ndarray:
    """
    Parameters
    ----------
    zarr_path  : path to preprocessed signals array, shape (N, T)
    lag        : AR lag — pairs x(t) with x(t - lag)
    chunk_size : number of time steps loaded per iteration

    Returns
    -------
    A : np.ndarray, shape (N, N)
        Estimated connectivity matrix (dense, no regularisation).
    """
    signals = zarr.open(zarr_path, mode="r")
    N, T = signals.shape

    # ------------------------------------------------------------------ #
    # Accumulators                                                         #
    # ------------------------------------------------------------------ #
    C_xx_raw = np.zeros((N, N))   # raw (uncentred) auto-covariance
    C_yx_raw = np.zeros((N, N))   # raw (uncentred) cross-covariance
    s_prev   = np.zeros(N)        # column sum of x_{t-1}
    s_now    = np.zeros(N)        # column sum of x_t
    n        = 0                  # total number of paired samples

    n_chunks = (T - lag) // chunk_size

    # ------------------------------------------------------------------ #
    # Single pass: accumulate raw cross-products and column sums          #
    # ------------------------------------------------------------------ #
    for k in range(n_chunks):
        t0 = k * chunk_size
        t1 = t0 + chunk_size

        x_prev = np.asarray(signals[:, t0       : t1      ])  # (N, chunk_size)
        x_now  = np.asarray(signals[:, t0 + lag : t1 + lag])  # (N, chunk_size)

        C_xx_raw += x_prev @ x_prev.T   # (N, N)
        C_yx_raw += x_now  @ x_prev.T   # (N, N)

        s_prev   += x_prev.sum(axis=1)  # (N,)
        s_now    += x_now .sum(axis=1)  # (N,)
        n        += x_prev.shape[1]

    # ------------------------------------------------------------------ #
    # Mean-centring correction (equivalent to sklearn fit_intercept=True) #
    #                                                                      #
    #   C_xx_c = C_xx_raw - (1/n) * s_prev ⊗ s_prev                      #
    #   C_yx_c = C_yx_raw - (1/n) * s_now  ⊗ s_prev                      #
    # ------------------------------------------------------------------ #
    mu_prev  = s_prev / n                           # per-neuron mean of x_{t-1}
    mu_now   = s_now  / n                           # per-neuron mean of x_t

    C_xx_c = C_xx_raw - n * np.outer(mu_prev, mu_prev)
    C_yx_c = C_yx_raw - n * np.outer(mu_now,  mu_prev)

    # ------------------------------------------------------------------ #
    # Solve the normal equations on the centred matrices                  #
    #                                                                      #
    #   C_yx_c = A @ C_xx_c  =>  A = C_yx_c @ inv(C_xx_c)               #
    # ------------------------------------------------------------------ #
    A = np.linalg.solve(C_xx_c.T, C_yx_c.T).T
    return A
