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
    on_checkpoint=None,
):
    """Stream calcium -> feed in chunks from a *run* BrunelNetwork (densify=False)
    and accumulate moments, snapshotting at each checkpoint (in samples).

    Returns (moments, tau_est, mean_rate_hz) where moments maps
    checkpoint_samples -> (Cxx, Cyx)."""
    N = net.N
    T_total = int(round(net.sim_time / dt))
    # Spike events dominate RAM at high firing rates: a 5e6 ms recording of 12500
    # neurons at ~60 Hz is ~3.7e9 events, which at int64+float64+int64 is ~90 GB
    # (this OOM'd a 120 GB node). Store the index as int16 (N < 32767) and the
    # time as an int32 SAMPLE number, and drop the float64 times -> ~6 bytes/event.
    # Then sort by time once so each chunk can be located with searchsorted,
    # instead of scanning every event with a boolean mask per chunk (which was
    # O(n_events * n_chunks) and the main reason streaming crawled).
    idx, times = net.get_spike_events()
    n_events = len(times)
    mean_rate = n_events / (N * net.sim_time / 1000.0)
    times_samp = np.round(times / dt).astype(np.int32)
    del times
    idx = idx.astype(np.int16) if N <= 32767 else idx.astype(np.int32)
    order = np.argsort(times_samp, kind="stable")
    times_samp = times_samp[order]
    idx = idx[order]
    del order

    # float32 throughout the chunk pipeline: ~2x less memory traffic and
    # noticeably faster noise/AR at large N, with no measurable effect on the
    # downstream metrics (the moments are reduced in float64).
    alpha = float(np.exp(-dt / tau))
    b = np.array([1.0], dtype=np.float32)
    a = np.array([1.0, -alpha], dtype=np.float32)
    zi = np.zeros((N, 1), dtype=np.float32)
    rng = np.random.default_rng(seed)
    f32 = np.float32

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
        # sparse -> dense spikes for this chunk only (events are time-sorted, so
        # the chunk's events are a contiguous slice — no full-array scan)
        spk = np.zeros((N, L), dtype=np.float32)
        lo = np.searchsorted(times_samp, t0, side="left")
        hi = np.searchsorted(times_samp, t1, side="left")
        if hi > lo:
            np.add.at(spk, (idx[lo:hi].astype(np.intp),
                            (times_samp[lo:hi] - t0).astype(np.intp)), f32(1.0))
        # calcium via AR(1) lfilter, carrying state zi across chunks
        # (standard_normal(dtype=f32)*sigma is markedly faster than normal(0,sigma))
        inp = spk
        inp *= f32(amplitude)
        inp += rng.standard_normal((N, L), dtype=np.float32) * f32(sigma_intra)
        C, zi = lfilter(b, a, inp, axis=1, zi=zi)
        del inp
        F = C
        F += rng.standard_normal((N, L), dtype=np.float32) * f32(sigma_extra)  # fluorescence
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
            Cxx_cp, Cyx_cp = acc.snapshot()
            if on_checkpoint is not None:
                # solve + persist immediately, so a wall-clock timeout later
                # still leaves the earlier checkpoints on disk
                on_checkpoint(checkpoints[ci], Cxx_cp, Cyx_cp)
            else:
                results[checkpoints[ci]] = (Cxx_cp, Cyx_cp)
            ci += 1

    # snapshot at full length for any checkpoints beyond T_total
    while ci < len(checkpoints):
        Cxx_cp, Cyx_cp = acc.snapshot()
        if on_checkpoint is not None:
            on_checkpoint(checkpoints[ci], Cxx_cp, Cyx_cp)
        else:
            results[checkpoints[ci]] = (Cxx_cp, Cyx_cp)
        ci += 1
    return results, tau_est, mean_rate
