"""
Calcium signal simulator.

Converts binary spike trains into fluorescence traces using a first-order
linear AR(1) model (single-exponential decay):

    C[t] = alpha * C[t-1] + A * S[t] + noise_intra[t]
    F[t] = C[t] + noise_extra[t]

where
    alpha  = exp(-dt / tau)   — discrete decay factor
    S[t]   — spike train (1 at spike times, 0 elsewhere)
    A      — spike-triggered calcium amplitude
    noise_intra ~ N(0, sigma_intra)   — intracellular/shot noise
    noise_extra ~ N(0, sigma_extra)   — recording/extracellular noise

Usage
-----
    from calcium_ar.simulation import simulate_calcium

    # spikes: (N, T) binary array from BrunelNetwork
    fluorescence = simulate_calcium(spikes, tau=100.0, dt=0.1)
"""

import numpy as np


def simulate_calcium(
    spikes: np.ndarray,
    tau: float = 100.0,
    dt: float = 0.1,
    amplitude: float = 1.0,
    sigma_intra: float = 0.01,
    sigma_extra: float = 0.05,
    warmup: int = 0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate calcium fluorescence from spike trains.

    Parameters
    ----------
    spikes : np.ndarray, shape (N, T) or (T,)
        Binary spike train matrix. 1 at spike times, 0 elsewhere.
    tau : float
        Calcium decay time constant in the same units as dt (default ms).
    dt : float
        Time step (ms). Must match the resolution used to generate spikes.
    amplitude : float
        Calcium transient amplitude per spike (ΔF/F units).
    sigma_intra : float
        Std of intracellular (shot) noise added at each time step.
    sigma_extra : float
        Std of extracellular/recording noise added to the final trace.
    warmup : int
        Number of initial time steps to discard so the signal reaches
        statistical stationarity before the output window.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    fluorescence : np.ndarray, same shape as spikes
        Simulated calcium fluorescence traces (ΔF/F).
    """
    rng = np.random.default_rng(seed)

    spikes = np.asarray(spikes, dtype=float)
    squeeze = spikes.ndim == 1
    if squeeze:
        spikes = spikes[np.newaxis, :]   # (1, T)

    N, T = spikes.shape
    alpha = np.exp(-dt / tau)

    # Pre-allocate
    C = np.zeros((N, T))

    # Intracellular noise added at each step
    noise_intra = rng.normal(0.0, sigma_intra, size=(N, T))

    # Forward pass — vectorised over neurons, sequential over time
    C[:, 0] = amplitude * spikes[:, 0] + noise_intra[:, 0]
    for t in range(1, T):
        C[:, t] = alpha * C[:, t - 1] + amplitude * spikes[:, t] + noise_intra[:, t]

    # Extracellular recording noise
    noise_extra = rng.normal(0.0, sigma_extra, size=(N, T))
    F = C + noise_extra

    if warmup > 0:
        F = F[:, warmup:]

    if squeeze:
        return F[0]
    return F
