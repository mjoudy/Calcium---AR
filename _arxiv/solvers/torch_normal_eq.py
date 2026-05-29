"""
GPU-accelerated chunked Ridge solver (PyTorch).

Mathematically identical to chunked_ridge.solve — accumulates C_xx and C_yx
on the GPU using torch matrix multiplications, then solves the small (N x N)
system with torch.linalg.solve (dispatches to cuSOLVER on GPU, LAPACK on CPU).

Why use this over numpy chunked_ridge?
  - Each chunk multiply runs on thousands of CUDA cores simultaneously
  - ~10–100x faster per chunk than numpy on CPU
  - Falls back to CPU automatically when no GPU is available

RAM cost : same as chunked_ridge — only one chunk + two N x N matrices
           (all on GPU VRAM, which is typically 8–80 GB on HPC nodes)
"""

import numpy as np
import zarr
import torch


def solve(zarr_path: str, lag: int = 10, lam: float = 1.0,
          chunk_size: int = 10_000, device: str = None) -> np.ndarray:
    """
    Parameters
    ----------
    zarr_path  : path to preprocessed signals array, shape (N, T)
    lag        : AR lag — pairs x(t) with x(t - lag)
    lam        : L2 regularization strength λ  (0 = plain OLS)
    chunk_size : number of time steps loaded per iteration
    device     : 'cuda', 'cpu', or None (auto-detects GPU)

    Returns
    -------
    A : np.ndarray, shape (N, N)
        Estimated connectivity matrix, returned as numpy array on CPU.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    signals = zarr.open(zarr_path, mode="r")
    N, T = signals.shape

    C_xx = torch.zeros(N, N, dtype=torch.float64, device=device)
    C_yx = torch.zeros(N, N, dtype=torch.float64, device=device)

    n_chunks = (T - lag) // chunk_size

    for k in range(n_chunks):
        t0 = k * chunk_size
        t1 = t0 + chunk_size

        # Load from disk → numpy → GPU (only chunk_size columns at a time)
        x_prev = torch.from_numpy(np.asarray(signals[:, t0       : t1      ])).to(device)
        x_now  = torch.from_numpy(np.asarray(signals[:, t0 + lag : t1 + lag])).to(device)

        C_xx += x_prev @ x_prev.T
        C_yx += x_now  @ x_prev.T

    C_xx_reg = C_xx + lam * torch.eye(N, dtype=torch.float64, device=device)

    # torch.linalg.solve uses cuSOLVER on GPU, LAPACK on CPU
    A = torch.linalg.solve(C_xx_reg, C_yx.T).T

    return A.cpu().numpy()
