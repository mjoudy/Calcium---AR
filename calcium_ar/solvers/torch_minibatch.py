"""
Sparse connectivity inference via mini-batch gradient descent (PyTorch).

Minimises the L1-regularised (Lasso) objective:

    L(A) = ||X_t - A X_{t-1}||_F^2  +  λ ||A||_1

The L1 term drives weak connection weights to exactly zero, producing a
sparse A that matches the biology of real neural circuits.

Why gradient descent instead of the normal equations?
  - The L1 penalty has no closed-form solution (non-differentiable at 0)
  - Gradient descent processes one chunk at a time → RAM = one chunk only
  - This is the only Lasso-equivalent method that runs on a laptop

How it works per iteration:
    1. Load chunk (x_prev, x_now) from disk
    2. Forward:  pred = A @ x_prev
    3. Loss:     reconstruction error + λ * sum(|A|)
    4. Backward: PyTorch auto-differentiates dL/dA
    5. Step:     Adam updates A

Multiple passes (epochs) over the full dataset are needed for convergence.
Each epoch costs ~2x the I/O and compute of one chunked OLS pass.
"""

import numpy as np
import zarr
import torch


def solve(zarr_path: str, lag: int = 10, lam: float = 1e-3,
          chunk_size: int = 10_000, n_epochs: int = 10,
          lr: float = 1e-3, device: str = None,
          on_epoch=None) -> np.ndarray:
    """
    Parameters
    ----------
    zarr_path  : path to preprocessed signals array, shape (N, T)
    lag        : AR lag — pairs x(t) with x(t - lag)
    lam        : L1 regularization strength λ — larger = sparser A
    chunk_size : number of time steps per mini-batch
    n_epochs   : number of full passes over the dataset
    lr         : Adam learning rate
    device     : 'cuda', 'cpu', or None (auto-detects GPU)

    Returns
    -------
    A : np.ndarray, shape (N, N)
        Estimated sparse connectivity matrix.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    signals = zarr.open(zarr_path, mode="r")
    N, T = signals.shape

    # A is the only learnable parameter — shape (N, N)
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

            # Forward pass
            residual = x_now - A @ x_prev            # (N, chunk_size)
            loss = (residual ** 2).sum()              # reconstruction
            loss = loss + lam * A.abs().sum()         # L1 penalty

            # Backward pass — PyTorch computes dL/dA automatically
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1:>3}/{n_epochs}  loss={epoch_loss:.6f}")
        if on_epoch is not None:
            on_epoch(epoch, epoch_loss)

    return A.detach().cpu().numpy()
