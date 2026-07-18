"""
Streaming feed -> moment accumulation for very long recordings.

The five wrap-up estimators all reduce to the lag-pair second moments
(Cxx, Cyx). For long T the full (N, T) feed will not fit in RAM, so we build it
in time-chunks (sparse spikes -> calcium via AR lfilter -> deconvolved feed) and
accumulate the moments incrementally with O(N^2) RAM. A MomentAccumulator can be
snapshotted at several checkpoints, so ONE long simulation yields a whole
data-length sweep.

Correctness: MomentAccumulator reproduces exactly (up to float error) the
centered moments that calcium_ar/solvers/fista.solve accumulates from the full
feed -- see scripts/validate_streaming.py.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from ..preprocessing.signal_utils import smooth_signal
from ..preprocessing.feed_reconstruction import reconstruct_feed
from ..preprocessing.tau_estimation import estimate_tau_robust


class MomentAccumulator:
    """Accumulate centered lag-pair second moments (Cxx, Cyx) over feed chunks.

    Feed the deconvolved feed in consecutive time-chunks via add(); a rolling
    tail of `lag` columns stitches lag-pairs across chunk boundaries with no gap
    or double-count. snapshot() returns the per-sample centered (Cxx, Cyx) for
    all feed seen so far."""

    def __init__(self, N: int, lag: int):
        self.N = N
        self.lag = lag
        self.Cxx_raw = np.zeros((N, N))
        self.Cyx_raw = np.zeros((N, N))
        self.s_prev = np.zeros(N)
        self.s_now = np.zeros(N)
        self.n = 0
        self.tail = np.zeros((N, 0))

    def add(self, feed_chunk: np.ndarray) -> None:
        buf = np.concatenate([self.tail, feed_chunk], axis=1)
        if buf.shape[1] <= self.lag:
            self.tail = buf
            return
        xp = buf[:, :-self.lag]
        xn = buf[:, self.lag:]
        self.Cxx_raw += xp @ xp.T
        self.Cyx_raw += xn @ xp.T
        self.s_prev += xp.sum(1)
        self.s_now += xn.sum(1)
        self.n += xp.shape[1]
        self.tail = buf[:, -self.lag:].copy()

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        if self.n == 0:
            raise RuntimeError("no pairs accumulated yet")
        mu_p = self.s_prev / self.n
        mu_n = self.s_now / self.n
        Cxx = (self.Cxx_raw - self.n * np.outer(mu_p, mu_p)) / self.n
        Cyx = (self.Cyx_raw - self.n * np.outer(mu_n, mu_p)) / self.n
        return Cxx, Cyx


class TorchMomentAccumulator:
    """GPU version of MomentAccumulator. The heavy lag-pair matmuls run on the
    device in float32 (fast on L40S); the reductions accumulate in float64 for
    stability across many chunks. snapshot() returns numpy float64 moments (moved
    to CPU) so the one-off OLS/Ridge solve can be done exactly on the CPU.

    Same interface as MomentAccumulator, so stream_moments() is device-agnostic."""

    def __init__(self, N: int, lag: int, device: str = "cuda"):
        import torch
        self.torch = torch
        self.N, self.lag, self.device = N, lag, device
        self.Cxx_raw = torch.zeros(N, N, dtype=torch.float64, device=device)
        self.Cyx_raw = torch.zeros(N, N, dtype=torch.float64, device=device)
        self.s_prev = torch.zeros(N, dtype=torch.float64, device=device)
        self.s_now = torch.zeros(N, dtype=torch.float64, device=device)
        self.n = 0
        self.tail = torch.zeros(N, 0, dtype=torch.float32, device=device)

    def add(self, feed_chunk) -> None:
        t = self.torch
        if isinstance(feed_chunk, t.Tensor):          # already deconvolved on device
            fc = feed_chunk.to(dtype=t.float32)
        else:
            fc = t.as_tensor(feed_chunk, dtype=t.float32, device=self.device)
        buf = t.cat([self.tail, fc], dim=1)
        if buf.shape[1] <= self.lag:
            self.tail = buf
            return
        xp = buf[:, :-self.lag]
        xn = buf[:, self.lag:]
        self.Cxx_raw += (xp @ xp.T).double()
        self.Cyx_raw += (xn @ xp.T).double()
        self.s_prev += xp.sum(1).double()
        self.s_now += xn.sum(1).double()
        self.n += xp.shape[1]
        self.tail = buf[:, -self.lag:].clone()

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        if self.n == 0:
            raise RuntimeError("no pairs accumulated yet")
        t = self.torch
        mu_p = self.s_prev / self.n
        mu_n = self.s_now / self.n
        Cxx = (self.Cxx_raw - self.n * t.outer(mu_p, mu_p)) / self.n
        Cyx = (self.Cyx_raw - self.n * t.outer(mu_n, mu_p)) / self.n
        return Cxx.cpu().numpy(), Cyx.cpu().numpy()


def stream_moments(
    net,
    *,
    dt: float,
    lag: int,
    tau: float,
    amplitude: float,
    sigma_intra: float,
    sigma_extra: float,
    smooth_win: int,
    tau_method: str,
    checkpoints_samples: list[int],
    chunk_samples: int = 100_000,
    seed: int = 1,
    device: str | None = None,
):
    """Stream calcium -> feed in chunks from a *run* BrunelNetwork (densify=False)
    and accumulate moments, snapshotting at each checkpoint (in samples).

    Returns (moments, tau_est, mean_rate_hz) where moments maps
    checkpoint_samples -> (Cxx, Cyx)."""
    N = net.N
    T_total = int(round(net.sim_time / dt))
    idx, times = net.get_spike_events()
    times_samp = np.round(times / dt).astype(np.int64)
    mean_rate = len(times) / (N * net.sim_time / 1000.0)

    alpha = float(np.exp(-dt / tau))
    b, a = [1.0], [1.0, -alpha]
    zi = np.zeros((N, 1))
    rng = np.random.default_rng(seed)

    acc = (TorchMomentAccumulator(N, lag, device=device) if device
           else MomentAccumulator(N, lag))

    # GPU deconvolution: Savitzky-Golay smoothing/derivative are fixed
    # convolutions, so on the device we run them as conv1d (matches scipy on
    # interior points to machine precision). Kernels depend only on the window;
    # tau enters when it is estimated on the first chunk. This removes the
    # CPU-side deconvolution bottleneck that dominates at large N.
    if device:
        import torch
        from torch.nn.functional import conv1d, pad as _pad
        from scipy.signal import savgol_coeffs
        _half = smooth_win // 2
        _ks = torch.tensor(savgol_coeffs(smooth_win, 3, deriv=0, delta=dt, use="dot"),
                           dtype=torch.float32, device=device).reshape(1, 1, smooth_win)
        _kd = torch.tensor(savgol_coeffs(smooth_win, 3, deriv=1, delta=dt, use="dot"),
                           dtype=torch.float32, device=device).reshape(1, 1, smooth_win)
        _tau_t = None

    checkpoints = sorted(checkpoints_samples)
    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    ci = 0
    tau_est = None

    t0 = 0
    while t0 < T_total:
        t1 = min(t0 + chunk_samples, T_total)
        L = t1 - t0
        # sparse -> dense spikes for this chunk only
        spk = np.zeros((N, L))
        m = (times_samp >= t0) & (times_samp < t1)
        np.add.at(spk, (idx[m], times_samp[m] - t0), 1.0)
        # calcium via AR(1) lfilter, carrying state zi across chunks
        inp = amplitude * spk + rng.normal(0.0, sigma_intra, (N, L))
        C, zi = lfilter(b, a, inp, axis=1, zi=zi)
        F = C + rng.normal(0.0, sigma_extra, (N, L))          # fluorescence
        if tau_est is None:
            tau_est = estimate_tau_robust(F, window_length=smooth_win,
                                          method=tau_method, dt=dt)
        # deconvolve chunk -> feed (per-chunk; boundary error negligible for
        # large chunks). GPU: conv1d on device; CPU: scipy savgol.
        if device:
            if _tau_t is None:
                _tau_t = torch.as_tensor(np.atleast_1d(tau_est), dtype=torch.float32,
                                         device=device).reshape(-1, 1)
            Ft = torch.as_tensor(F, dtype=torch.float32, device=device).unsqueeze(1)
            Ft = _pad(Ft, (_half, _half), mode="replicate")   # edge-safe padding
            sm = conv1d(Ft, _ks).squeeze(1)
            dv = conv1d(Ft, _kd).squeeze(1)
            acc.add(dv + sm / _tau_t)
        else:
            smooth = smooth_signal(F, window_length=smooth_win, deriv=0, delta=dt)
            deriv = smooth_signal(F, window_length=smooth_win, deriv=1, delta=dt)
            acc.add(reconstruct_feed(smooth, deriv, tau_est))
        t0 = t1
        while ci < len(checkpoints) and t1 >= checkpoints[ci]:
            results[checkpoints[ci]] = acc.snapshot()
            ci += 1

    # snapshot at full length for any checkpoints beyond T_total
    while ci < len(checkpoints):
        results[checkpoints[ci]] = acc.snapshot()
        ci += 1
    return results, tau_est, mean_rate
