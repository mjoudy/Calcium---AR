"""
Chunked FISTA solver for Elastic Net (L1 + L2) connectivity inference.

Minimises:
    L(A) = (1/2T) ||X_t - A X_{t-1}||_F^2
             +  lam_l1 * ||A||_1
             +  (lam_l2/2) * ||A||_F^2

Algorithm: FISTA (Beck & Teboulle 2009)
  - Gradient step on the smooth part (quadratic + L2) with Nesterov momentum
  - Proximal step = element-wise soft-thresholding for the L1 part
  - Step size = 1/L, where L = λ_max(C_xx_c/T) + lam_l2 is the Lipschitz
    constant of the smooth gradient — computed automatically, no tuning needed
  - Convergence rate: O(1/k²)

Gram matrices are accumulated in chunks (same pattern as chunked_ols/ridge),
so RAM cost is O(N²) regardless of T.

Special cases:
  lam_l1 = 0, lam_l2 > 0  →  Ridge (matches chunked_ridge for n_iter → ∞,
                               but chunked_ridge is exact and faster)
  lam_l1 > 0, lam_l2 = 0  →  Lasso  (same objective as sklearn_lasso)
  lam_l1 = 0, lam_l2 = 0  →  OLS    (use chunked_ols for the exact solution)
"""

import numpy as np
import zarr


def solve(
    zarr_path: str,
    lag: int = 10,
    lam_l1: float = 1e-3,
    lam_l2: float = 1e-3,
    n_iter: int = 500,
    chunk_size: int = 10_000,
) -> np.ndarray:
    """
    Parameters
    ----------
    zarr_path : path to preprocessed signals array, shape (N, T)
    lag       : AR lag in samples
    lam_l1    : L1 regularisation strength (sparsity)
    lam_l2    : L2 regularisation strength (stability / handles correlated neurons)
    n_iter    : number of FISTA iterations (500 is ample for N ≤ 1000)
    chunk_size: time steps loaded per accumulation pass

    Returns
    -------
    A : np.ndarray, shape (N, N)
        Sparse connectivity matrix.  A[i, j] = inferred weight from j to i.
    """
    signals = zarr.open(zarr_path, mode="r")
    N, T = signals.shape

    # ------------------------------------------------------------------ #
    # Accumulate gram matrices in chunks (O(N²) RAM)                      #
    # ------------------------------------------------------------------ #
    C_xx_raw = np.zeros((N, N))
    C_yx_raw = np.zeros((N, N))
    s_prev   = np.zeros(N)
    s_now    = np.zeros(N)
    n        = 0

    n_full   = (T - lag) // chunk_size
    n_chunks = n_full + (1 if (T - lag) % chunk_size else 0)

    for k in range(n_chunks):
        t0 = k * chunk_size
        t1 = min(t0 + chunk_size, T - lag)
        x_prev = np.asarray(signals[:, t0       : t1      ])
        x_now  = np.asarray(signals[:, t0 + lag : t1 + lag])
        C_xx_raw += x_prev @ x_prev.T
        C_yx_raw += x_now  @ x_prev.T
        s_prev   += x_prev.sum(axis=1)
        s_now    += x_now .sum(axis=1)
        n        += x_prev.shape[1]

    # ------------------------------------------------------------------ #
    # Mean-centring correction                                             #
    # ------------------------------------------------------------------ #
    mu_prev = s_prev / n
    mu_now  = s_now  / n
    C_xx_c  = C_xx_raw - n * np.outer(mu_prev, mu_prev)
    C_yx_c  = C_yx_raw - n * np.outer(mu_now,  mu_prev)

    # Normalise to per-sample scale so lam values are T-invariant
    XtX = C_xx_c / n
    XtY = C_yx_c / n

    # ------------------------------------------------------------------ #
    # Step size: 1 / Lipschitz constant of smooth gradient                #
    # smooth part = (1/2)||A XtX^{1/2} - XtY XtX^{-1/2}||_F^2 + lam_l2/2||A||^2
    # L = λ_max(XtX) + lam_l2                                            #
    # ------------------------------------------------------------------ #
    L    = float(np.linalg.eigvalsh(XtX).max()) + lam_l2
    step = 1.0 / L

    # ------------------------------------------------------------------ #
    # FISTA iterate                                                        #
    # ------------------------------------------------------------------ #
    A      = np.zeros((N, N))
    A_prev = np.zeros((N, N))
    t_k    = 1.0

    for _ in range(n_iter):
        # Beck-Teboulle t sequence: t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2))

        # Nesterov momentum point
        y = A + ((t_k - 1.0) / t_next) * (A - A_prev)

        # Gradient of smooth part at y
        grad = y @ XtX - XtY + lam_l2 * y

        # Gradient step + soft-threshold (proximal operator of L1)
        A_prev = A
        u = y - step * grad
        A = np.sign(u) * np.maximum(np.abs(u) - step * lam_l1, 0.0)

        t_k = t_next

    return A
