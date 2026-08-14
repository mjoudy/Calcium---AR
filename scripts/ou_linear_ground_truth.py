"""
Analytic linear ground truth: exact stationary moments of a multivariate
Ornstein-Uhlenbeck process built on the SAME connectivity as an existing
cached LIF network, used to test whether shared-input false positives survive
when the generative process is genuinely, exactly linear (no spiking
threshold/reset nonlinearity at all) and the moments have NO finite-sample
noise (solved analytically, not estimated from simulated timesteps).

Model:  dx(t) = A x(t) dt + dW(t),   A = -I/tau + s*G,   G = adj_true.T
  - G[i,k] = adj_true[k,i]: coupling FROM source k TO target i (standard
    drift-matrix convention: row = target, column = source). adj_true itself
    is left untouched on disk (still [source,target]) so the existing
    fig_way2*.py / attribution scripts, which already transpose it
    themselves, work unchanged.
  - s rescales the coupling to a stable regime (spectral margin below the
    leak rate 1/tau) -- this changes overall coupling STRENGTH, not sign or
    sparsity pattern, so it doesn't affect which pairs count as true edges.
  - Noise is isotropic unit white noise, Q = I.

Stationary covariance solves the continuous Lyapunov equation
    A @ Cxx + Cxx @ A.T + Q = 0
and the lag-l cross-covariance is
    Cyx(l) = expm(A * l) @ Cxx
both exact, closed-form -- no simulation, no sampling noise. Saves
Cxx.npy/Cyx.npy/adj_true.npy in the same format as best_moments/*, so they
drop straight into fig_way2.py / fig_way2_motifs.py / any downstream script
with --data pointing at the output dir.

Usage:
  python scripts/ou_linear_ground_truth.py \
      --adj ~/calcium_results/best_moments/n1250r4/adj_true.npy \
      --out ~/calcium_results/ou_moments/n1250_linear \
      --tau 1.0 --lag-frac 0.1 --margin 0.2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import solve_continuous_lyapunov, expm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adj", required=True, help="path to an existing adj_true.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=1.0, help="leak/decay time constant")
    ap.add_argument("--lag-frac", type=float, default=0.1,
                     help="regression lag as a fraction of tau")
    ap.add_argument("--margin", type=float, default=0.2,
                     help="stability safety margin (fraction of leak rate 1/tau)")
    args = ap.parse_args()

    adj = np.load(args.adj).astype(np.float64)
    N = adj.shape[0]
    np.fill_diagonal(adj, 0.0)
    G = adj.T.copy()                                   # (target, source) drift convention

    print(f"N={N}  computing eigenvalues of G for stability scaling...")
    eig = np.linalg.eigvals(G)
    rho = eig.real.max()
    print(f"max Re(eig(G)) = {rho:.4f}")

    tau = args.tau
    leak = 1.0 / tau
    if rho > 0:
        s = (1.0 - args.margin) * leak / rho
    else:
        s = 1.0
    A_mat = -leak * np.eye(N) + s * G
    abscissa = np.linalg.eigvals(A_mat).real.max()
    print(f"coupling scale s={s:.4g}  drift-matrix spectral abscissa={abscissa:.4f} "
          f"(must be < 0 for stability; leak rate is -{leak:.4f})")
    assert abscissa < 0, "A_mat is not stable -- increase --margin"

    Q = np.eye(N)
    print("solving continuous Lyapunov equation for stationary Cxx ...")
    Cxx = solve_continuous_lyapunov(A_mat, -Q)
    Cxx = 0.5 * (Cxx + Cxx.T)                           # symmetrize (numerical)

    lag = args.lag_frac * tau
    print(f"lag = {lag:.4g} (= {args.lag_frac} * tau)  computing Cyx = expm(A*lag) @ Cxx ...")
    Cyx = expm(A_mat * lag) @ Cxx

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "Cxx.npy", Cxx)
    np.save(out / "Cyx.npy", Cyx)
    np.save(out / "adj_true.npy", adj)                 # original, unscaled, [source,target]
    meta = dict(tau=tau, lag=lag, lag_frac=args.lag_frac, coupling_scale=s,
                margin=args.margin, spectral_abscissa=float(abscissa),
                rho_G=float(rho), source_adj=str(args.adj))
    np.save(out / "meta.npy", meta, allow_pickle=True)
    print(f"wrote {out}/Cxx.npy Cyx.npy adj_true.npy meta.npy")

    # quick sanity check: does plain single-lag OLS recover real edges at all?
    A_reg = Cyx @ np.linalg.inv(Cxx + 1e-9 * np.eye(N))
    np.fill_diagonal(A_reg, 0.0)
    aa = np.abs(A_reg)
    truth = (adj.T != 0)                                # true in A's orientation
    tau_thr = np.quantile(aa[~np.eye(N, dtype=bool)], 0.90)
    pred = aa > tau_thr
    off = ~np.eye(N, dtype=bool)
    tp = (pred & truth & off).sum(); fp = (pred & ~truth & off).sum()
    prec = tp / max(tp + fp, 1)
    print(f"sanity check (top-10% |A_reg|): precision={prec:.3f}  "
          f"mean|A| true edges={aa[truth & off].mean():.4g}  "
          f"mean|A| non-edges={aa[~truth & off].mean():.4g}")


if __name__ == "__main__":
    main()
