"""
Post-processing for inferred connectivity: sign (Dale) and magnitude (balance).

These are *optional* steps applied after a solver, kept separate from the core
runner. They were previously copy-pasted across several analysis scripts; this
module is the single canonical home.

Two jobs:
- **Dale** — fix the signs. Either a hard cleanup (`dale_cleanup`) or, better, a
  sign-constrained re-solve (`dale_fista`) that forces every neuron's outgoing
  weights to one sign (all excitatory or all inhibitory).
- **Balance** — fix the magnitude. `rescale_balance_nz` rescales inhibitory
  weights so the E/I strength ratio matches `g`, estimated from the network
  balance (`balance_g`) using only observable firing rates + population sizes.

All functions are unsupervised: they never use the ground-truth network.
"""

from __future__ import annotations

import numpy as np


def strongest_entry_types(A: np.ndarray) -> np.ndarray:
    """Unsupervised type per source column = sign of its largest-magnitude entry.

    Returns an (N,) array of +1 (excitatory) / -1 (inhibitory) per neuron.
    """
    N = A.shape[0]
    t = np.ones(N)
    for j in range(N):
        col = A[:, j].copy()
        col[j] = 0.0
        t[j] = np.sign(col[np.argmax(np.abs(col))]) or 1.0
    return t


def dale_cleanup(A: np.ndarray) -> np.ndarray:
    """Hard Dale: per column, zero every entry whose sign disagrees with the
    column's inferred type. (Post-hoc cleanup — see dale_fista for the stronger
    in-solver version.)"""
    N = A.shape[0]
    B = A.copy()
    t = strongest_entry_types(A)
    for j in range(N):
        col = B[:, j]
        col[np.sign(col) != t[j]] = 0.0
        B[:, j] = col
    np.fill_diagonal(B, 0.0)
    return B


def dale_fista(
    X: np.ndarray,
    Y: np.ndarray,
    types: np.ndarray,
    lam1: float = 3e-3,
    lam2: float = 1e-3,
    n_iter: int = 800,
) -> np.ndarray:
    """Sign-constrained Elastic Net (Dale as regularization) via FISTA.

    Solves the AR regression again, but projects each column to its allowed sign
    (`types`) every iteration, so the result obeys Dale's law by construction.

    Parameters
    ----------
    X, Y : (N, M) centred lag pairs (predictors / targets).
    types : (N,) +1 / -1 sign allowed per source column (e.g. from
        strongest_entry_types on a first-pass solution).
    lam1, lam2 : L1 / L2 strengths.
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


def balance_g(types: np.ndarray, rates: np.ndarray) -> float:
    """Estimate the E/I weight ratio g from network balance:
    g ≈ (N_E·νE) / (N_I·νI), using observed per-neuron firing rates."""
    E = types > 0
    I = types < 0
    if E.sum() == 0 or I.sum() == 0 or rates[I].mean() == 0:
        return np.nan
    return (E.sum() * rates[E].mean()) / (I.sum() * rates[I].mean())


def rescale_balance_nz(A: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Balance rescale using NON-ZERO entries only.

    Lifts the inferred inhibitory columns so the I-vs-E median magnitude matches
    g (from balance_g). Uses non-zero entries so it composes with Dale/mixture
    steps (which zero many entries) instead of breaking on an all-zeros median.
    """
    N = A.shape[0]
    off = ~np.eye(N, dtype=bool)
    B = A.copy()
    t = strongest_entry_types(A)
    g = balance_g(t, rates)
    Ec, Ic = np.where(t > 0)[0], np.where(t < 0)[0]

    def med_nz(cols):
        mk = np.isin(np.arange(N), cols)[None, :] & off
        v = np.abs(A[mk])
        v = v[v > 1e-9]
        return np.median(v) if v.size else 0.0

    medE, medI = med_nz(Ec), med_nz(Ic)
    if medI > 1e-9 and medE > 0 and np.isfinite(g):
        B[:, Ic] = A[:, Ic] * (g * medE / medI)
    np.fill_diagonal(B, 0.0)
    return B
