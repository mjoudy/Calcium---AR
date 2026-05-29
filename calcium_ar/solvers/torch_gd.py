"""
Connectivity inference via gradient descent — no regularization.

Minimises the plain least-squares objective:

    L(A) = ||X_t - A X_{t-1}||_F^2

This is the same objective as OLS (normal equations), solved iteratively
with Adam instead of analytically.  Useful for isolating the effect of
the optimization method from the effect of regularization:

  OLS         — exact solution, L2 objective
  torch_gd    — iterative solution, same L2 objective   ← this file
  ridge       — exact solution, L2 + L2 penalty
  lasso       — iterative solution, L2 + L1 penalty
"""

import numpy as np
import zarr
import torch


def solve(zarr_path: str, lag: int = 10,
          chunk_size: int = 10_000, n_epochs: int = 100,
          lr: float = 1e-3, device: str = None,
          on_epoch=None) -> np.ndarray:
    """
    Parameters
    ----------
    zarr_path  : path to preprocessed signals array, shape (N, T)
    lag        : AR lag — pairs x(t) with x(t - lag)
    chunk_size : number of time steps per mini-batch
    n_epochs   : number of full passes over the dataset
    lr         : Adam learning rate
    device     : 'cuda', 'cpu', or None (auto-detects GPU)

    Returns
    -------
    A : np.ndarray, shape (N, N)
        Estimated connectivity matrix.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    signals = zarr.open(zarr_path, mode="r")
    N, T = signals.shape

    A = torch.zeros(N, N, dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([A], lr=lr)

    n_chunks = (T - lag) // chunk_size

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for k in range(n_chunks):
            t0 = k * chunk_size
            t1 = t0 + chunk_size

            x_prev = torch.from_numpy(np.asarray(signals[:, t0       : t1      ])).to(device)
            x_now  = torch.from_numpy(np.asarray(signals[:, t0 + lag : t1 + lag])).to(device)

            loss = ((x_now - A @ x_prev) ** 2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1:>3}/{n_epochs}  loss={epoch_loss:.6f}")
        if on_epoch is not None:
            on_epoch(epoch, epoch_loss)

    return A.detach().cpu().numpy()
