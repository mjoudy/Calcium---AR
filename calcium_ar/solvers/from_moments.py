"""
Solvers that work directly from the lag-pair second moments (Cxx, Cyx) instead
of the full feed. All five wrap-up methods reduce to these two N×N matrices, so
for very long recordings we accumulate the moments in a streaming pass (see
calcium_ar/experiments/streaming.py) and then solve here with O(N^2) RAM.

Convention matches the streaming/fista path:
    Cxx = centered  <x_{t}   x_{t}^T>   (per-sample, N×N)
    Cyx = centered  <x_{t+lag} x_{t}^T> (per-sample, N×N)
    A  ~  Cyx Cxx^{-1}    (A[i,j] = inferred weight j -> i)
"""

from __future__ import annotations

import numpy as np


def ols_from_moments(Cxx: np.ndarray, Cyx: np.ndarray, ridge: float = 1e-9) -> np.ndarray:
    """Ordinary least squares (tiny ridge for conditioning)."""
    N = Cxx.shape[0]
    A = Cyx @ np.linalg.inv(Cxx + ridge * np.eye(N))
    np.fill_diagonal(A, 0.0)
    return A


def fista_from_moments(Cxx: np.ndarray, Cyx: np.ndarray, lam_l1: float,
                       lam_l2: float, n_iter: int = 500) -> np.ndarray:
    """Elastic Net / Lasso (lam_l2=0) via FISTA — identical iteration to
    calcium_ar/solvers/fista.solve but on precomputed moments."""
    N = Cxx.shape[0]
    L = float(np.linalg.eigvalsh(Cxx).max()) + lam_l2
    step = 1.0 / L
    A = np.zeros((N, N)); A_prev = np.zeros((N, N)); t_k = 1.0
    for _ in range(n_iter):
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2))
        y = A + ((t_k - 1.0) / t_next) * (A - A_prev)
        grad = y @ Cxx - Cyx + lam_l2 * y
        A_prev = A
        u = y - step * grad
        A = np.sign(u) * np.maximum(np.abs(u) - step * lam_l1, 0.0)
        t_k = t_next
    np.fill_diagonal(A, 0.0)
    return A


def dale_from_moments(Cxx: np.ndarray, Cyx: np.ndarray, types: np.ndarray,
                      lam1: float, lam2: float, n_iter: int = 800) -> np.ndarray:
    """Sign-constrained Elastic Net (Dale) via FISTA — identical iteration to
    calcium_ar/solvers/dale_fista.dale_fista but on precomputed moments."""
    N = Cxx.shape[0]
    Lip = float(np.linalg.eigvalsh(Cxx)[-1]) + lam2
    thr = lam1 / Lip
    tcol = types[None, :]
    A = np.zeros((N, N)); Z = A.copy(); tk = 1.0
    for _ in range(n_iter):
        V = Z - (Z @ Cxx - Cyx + lam2 * Z) / Lip
        V = np.sign(V) * np.maximum(np.abs(V) - thr, 0.0)
        V = tcol * np.maximum(tcol * V, 0.0)
        tnew = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        Z = V + ((tk - 1) / tnew) * (V - A); A, tk = V, tnew
    np.fill_diagonal(A, 0.0)
    return A


def strongest_entry_types(A: np.ndarray) -> np.ndarray:
    """Per-neuron sign from its strongest outgoing (column) weight; +1 default."""
    N = A.shape[0]
    t = np.ones(N)
    for j in range(N):
        c = A[:, j].copy(); c[j] = 0.0
        if np.any(c):
            t[j] = np.sign(c[np.argmax(np.abs(c))]) or 1.0
    return t
