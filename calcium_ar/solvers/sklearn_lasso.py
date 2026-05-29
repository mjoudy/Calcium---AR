"""
Connectivity inference via sklearn Lasso (coordinate descent).

Minimises the L1-regularised least-squares objective:

    L(A) = (1 / 2T) * ||X_t - A X_{t-1}||_F^2  +  alpha * ||A||_1

Coordinate descent (sklearn's implementation) is the standard solver for L1
problems — it converges orders of magnitude faster than gradient descent
(Adam) because it optimises one weight at a time with an exact closed-form
update, exploiting the separability of the L1 penalty.

Design matrix approach
----------------------
The full (T-lag) × N matrix X_prev is loaded into RAM once, then N
independent Lasso regressions are solved in parallel (one per target neuron).
RAM cost: O(T × N).  Fine for T=10^5, N=100 (~80 MB); use with caution for
T=10^6 (~800 MB).

sklearn normalisation note
--------------------------
sklearn scales its loss by 1/(2T), so its alpha is NOT comparable to the lam
used in torch_minibatch (which uses the un-normalised loss).  Equivalent
mapping:  alpha_sklearn ≈ lam_torch / (2 * T).
A good empirical starting range for alpha here is 1e-4 … 1e-1; larger values
produce sparser A, smaller values approach the OLS solution.
"""

import numpy as np
import zarr
from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor


def solve(
    zarr_path: str,
    lag: int = 10,
    lam: float = 1e-3,
    chunk_size: int = 10_000,   # unused — kept for uniform API
    n_jobs: int = -1,
    max_iter: int = 5_000,
) -> np.ndarray:
    """
    Parameters
    ----------
    zarr_path : path to preprocessed signals array, shape (N, T)
    lag       : AR lag — pairs x(t) with x(t - lag)
    lam       : sklearn Lasso alpha (regularisation strength).
                Larger → sparser A.  Typical range: 1e-4 … 1e-1.
    chunk_size: ignored (full matrix is loaded); kept for API consistency
    n_jobs    : parallel jobs for MultiOutputRegressor (-1 = all cores)
    max_iter  : maximum coordinate-descent iterations per neuron

    Returns
    -------
    A : np.ndarray, shape (N, N)
        Estimated sparse connectivity matrix.
        A[i, j] = inferred weight from neuron j to neuron i.
    """
    signals = zarr.open(zarr_path, mode="r")
    N, T = signals.shape

    # Load full design matrix — O(T × N) RAM
    data   = np.asarray(signals, dtype=float)   # (N, T)
    x_prev = data[:, :T - lag].T                # (T-lag, N) — features
    x_now  = data[:, lag:].T                    # (T-lag, N) — targets

    # N independent Lasso regressions, one per target neuron, run in parallel
    # fit_intercept=True is equivalent to mean-centring (same as OLS/Ridge)
    model = MultiOutputRegressor(
        Lasso(alpha=lam, fit_intercept=True, max_iter=max_iter, tol=1e-4),
        n_jobs=n_jobs,
    )
    model.fit(x_prev, x_now)

    # estimators_[i].coef_ shape: (N,) = row i of A
    A = np.array([est.coef_ for est in model.estimators_])
    return A
