"""
Sparse connectivity inference using PyTorch nn.Linear (L1 regularized).

Mathematically identical to torch_minibatch — minimises:

    L(A) = ||X_t - A X_{t-1}||_F^2  +  λ ||A||_1

The difference is architectural: A is wrapped inside an nn.Module instead of
being a raw parameter tensor. This makes it straightforward to later extend to:
  - multiple lags  (stack several Linear layers, one per lag)
  - nonlinear dynamics  (add activation functions between layers)
  - constrained connectivity  (e.g. non-negative weights for excitatory-only)
  - mixed E/I architectures  (split neurons into subpopulations)

nn.Linear convention:
    output = input @ weight.T      where weight.shape = (out, in) = (N, N)

With input = x_prev.T  (shape: chunk x N, i.e. time steps as rows):
    output = x_prev.T @ A.T = (A @ x_prev).T = x_now.T

So model.linear.weight  IS  your connectivity matrix A directly.
"""

import numpy as np
import zarr
import torch
import torch.nn as nn


class _ConnectivityLayer(nn.Module):
    def __init__(self, N: int):
        super().__init__()
        self.linear = nn.Linear(N, N, bias=False, dtype=torch.float64)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x_prev_T: torch.Tensor) -> torch.Tensor:
        # x_prev_T : (chunk_size, N)  — time steps as rows
        # returns  : (chunk_size, N)  — predicted x_now, time steps as rows
        return self.linear(x_prev_T)


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
        Estimated sparse connectivity matrix (= model.linear.weight).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    signals = zarr.open(zarr_path, mode="r")
    N, T = signals.shape

    model = _ConnectivityLayer(N).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_chunks = (T - lag) // chunk_size

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for k in range(n_chunks):
            t0 = k * chunk_size
            t1 = t0 + chunk_size

            # nn.Linear expects (samples, features) = (chunk_size, N)
            x_prev_T = torch.from_numpy(np.asarray(signals[:, t0       : t1      ]).T).to(device)
            x_now_T  = torch.from_numpy(np.asarray(signals[:, t0 + lag : t1 + lag]).T).to(device)

            pred = model(x_prev_T)                            # (chunk_size, N)
            loss = ((pred - x_now_T) ** 2).sum()              # reconstruction
            loss = loss + lam * model.linear.weight.abs().sum()  # L1 penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1:>3}/{n_epochs}  loss={epoch_loss:.6f}")
        if on_epoch is not None:
            on_epoch(epoch, epoch_loss)

    return model.linear.weight.detach().cpu().numpy()   # (N, N) = A
