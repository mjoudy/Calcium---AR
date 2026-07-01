"""
Dale-regularized solver: sign-constrained Elastic Net via FISTA.

This is a *solver* (it estimates connectivity from the data), but a two-stage
one: it needs the per-neuron E/I types, which come from a first-pass estimate
(e.g. plain Elastic Net). Every iteration it projects each column to its allowed
sign, so the result obeys Dale's law by construction. Your experiments found this
in-solver Dale beats post-hoc sign cleanup.

Note the signature differs from the streaming solvers (which take a zarr path):
this one takes the centred lag pairs in memory, because it re-uses the second
moments across iterations.
"""

from __future__ import annotations

import numpy as np


def dale_fista(
    X: np.ndarray,
    Y: np.ndarray,
    types: np.ndarray,
    lam1: float = 3e-3,
    lam2: float = 1e-3,
    n_iter: int = 800,
) -> np.ndarray:
    """Sign-constrained Elastic Net (Dale as regularization) via FISTA.

    Parameters
    ----------
    X, Y : (N, M) centred lag pairs (predictors / targets).
    types : (N,) +1 (excitatory) / -1 (inhibitory) sign allowed per source
        column, e.g. from strongest_entry_types() on a first-pass solution.
    lam1, lam2 : L1 / L2 strengths.
    n_iter : number of FISTA iterations.
    """
    N, M = X.shape
    Cxx = (X @ X.T) / M
    Cyx = (Y @ X.T) / M
    Lip = np.linalg.eigvalsh(Cxx)[-1] + lam2
    thr = lam1 / Lip
    tcol = types[None, :]
    A = np.zeros((N, N))
    Z = A.copy()
    tk = 1.0
    for _ in range(n_iter):
        V = Z - (Z @ Cxx - Cyx + lam2 * Z) / Lip
        V = np.sign(V) * np.maximum(np.abs(V) - thr, 0.0)   # L1 soft-threshold
        V = tcol * np.maximum(tcol * V, 0.0)                # per-column sign projection
        tnew = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        Z = V + ((tk - 1) / tnew) * (V - A)
        A, tk = V, tnew
    np.fill_diagonal(A, 0.0)
    return A
