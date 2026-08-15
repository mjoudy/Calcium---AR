"""
Exact linear multivariate Hawkes process, built on the SAME connectivity as an
existing cached LIF/OU network, used to answer the professor's comment on
fig_linearity_way2: the OU arm has no spikes at all, so "deconvolution" was
never meaningfully applied there (its Cxx/Cyx come straight from the
continuous Lyapunov equation, see ou_linear_ground_truth.py). A Hawkes process
IS a genuine point process -- it has real spikes -- and it is exactly the
model Pernice, Staude, Cardanobile, Rotter (2011, PLoS Comput Biol) built their
motif-decomposition theory around, so it is a strictly better "linear ground
truth" than OU for this specific comparison.

Model (Pernice 2011 Methods, Eqs. 1-10):
    y_i(t) = [ y0 + sum_k (h_ik * s_k)(t) ]_+          (rectified rate)
    h_ik(t) = (G[i,k] / tau_syn) * exp(-t/tau_syn)      for t > 0, else 0
  where G[i,k] = adj_true[k,i] (source k -> target i, standard drift-matrix
  convention -- adj_true itself is left untouched on disk, [source,target],
  same as ou_linear_ground_truth.py) and G is rescaled by a stability factor s
  so that max(|eig(G)|) sits below 1 with a safety margin (Pernice's exact
  condition for the correlation series in Eq. 14 to converge -- NOTE this is
  the eigenvalue MAGNITUDE, unlike the OU script's real-part/abscissa
  condition, because a Hawkes G is a discrete-generator matrix, not a
  continuous-time drift matrix).
    y0 = 10 Hz for every neuron, tau_syn = 10 ms -- both taken directly from
  Pernice 2011's own simulated example (Methods / Fig. 2 caption) so the
  numbers are traceable to the paper, not arbitrary.

Simulation: exact event-driven thinning (Ogata 1981) exploiting the
exponential kernel -- between events the excitation state m(t) decays as
m(t)*exp(-dt/tau_syn) with no simulation time step needed; each spike of
neuron k adds G[:,k]/tau_syn to m. A neuron-wise upper bound on the intensity
over [t, t+dt] is y0 + max(0, m_i(t)) (proven in the script docstring below,
inline) which gives a tight, valid thinning envelope. No NEST needed -- this
is pure NumPy and, unlike an N x T dense simulation, memory cost is O(N +
n_events), not O(N*T), so it is safe to run locally.

Saves spike EVENTS (idx.npy, times_ms.npy) rather than a dense (N,T) spike
matrix -- feed straight into scripts/hawkes_to_moments.py, which mirrors
scripts/wrapup_run_stream.py's spike_events -> calcium -> deconvolution ->
streaming-moments path so the Hawkes arm goes through the EXACT SAME
observation pipeline as the LIF/PIF arms (unlike OU, which skips it).

Usage:
  python scripts/hawkes_ground_truth.py \\
      --adj ~/calcium_results/best_moments/n1250r4/adj_true.npy \\
      --out ~/calcium_results/hawkes_events/n1250 \\
      --sim-time 100000 --seed 1

  Smoke test (seconds, not minutes):
  python scripts/hawkes_ground_truth.py --adj <...> --out /tmp/hawkes_smoke \\
      --sim-time 2000 --n-max 50
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def build_G(adj: np.ndarray, margin: float) -> tuple[np.ndarray, dict]:
    """G[i,k] = adj[k,i] (target row, source col), rescaled so
    max(|eig(G)|) = 1 - margin. Returns (G_scaled, info)."""
    N = adj.shape[0]
    G_raw = adj.T.copy()
    eig = np.linalg.eigvals(G_raw)
    rho = float(np.abs(eig).max())
    target = 1.0 - margin
    s = target / rho if rho > 0 else 1.0
    G = s * G_raw
    rho_scaled = float(np.abs(np.linalg.eigvals(G)).max())
    assert rho_scaled < 1.0, "G is not in the convergent regime -- increase --margin"
    info = dict(N=N, rho_raw=rho, coupling_scale=s, rho_scaled=rho_scaled, margin=margin)
    return G, info


def simulate_hawkes(G: np.ndarray, y0_hz: float, tau_syn_ms: float, sim_time_ms: float,
                     seed: int, max_events: int | None = None,
                     progress_every: int = 500_000) -> tuple[np.ndarray, np.ndarray, dict]:
    """Exact thinning simulation of the linear multivariate Hawkes process.

    Returns (idx, times_ms, stats). idx/times_ms match
    BrunelNetwork.get_spike_events()'s format (0-indexed neuron, ms), just
    unsorted -- stream_moments sorts internally.

    Why the bound y0 + max(0, m_i(t)) is valid for s in [t, t+anything]:
    between events m decays as m(t)*exp(-(s-t)/tau) (same sign throughout, no
    new jumps). If m_i(t) >= 0, y0+m_i(s) is DEcreasing in s, so its sup on
    [t,inf) is the value at s=t, i.e. y0+m_i(t). If m_i(t) < 0, y0+m_i(s) is
    INcreasing in s toward y0 (never reaching it), so its sup is y0. Both
    cases are covered by y0 + max(0, m_i(t)), evaluated once at the start of
    each candidate draw -- a fresh, valid Ogata envelope every step.
    """
    N = G.shape[0]
    rng = np.random.default_rng(seed)
    inv_tau = 1.0 / tau_syn_ms
    y0 = y0_hz / 1000.0  # Hz -> spikes/ms, matches ms time unit used throughout

    m = np.zeros(N)
    t = 0.0
    idx_out: list[int] = []
    times_out: list[float] = []
    n_candidates = 0
    t0 = time.time()

    while True:
        pos = np.clip(m, 0.0, None)
        Lambda_star = N * y0 + pos.sum()
        dt_cand = rng.exponential(1.0 / Lambda_star)
        t_cand = t + dt_cand
        if t_cand > sim_time_ms:
            break
        n_candidates += 1

        decay = np.exp(-dt_cand * inv_tau)
        m_cand = m * decay
        lam = np.clip(y0 + m_cand, 0.0, None)
        Lambda_true = lam.sum()

        if rng.random() * Lambda_star <= Lambda_true:
            r = rng.random() * Lambda_true
            k = int(np.searchsorted(np.cumsum(lam), r))
            k = min(k, N - 1)
            idx_out.append(k)
            times_out.append(t_cand)
            m = m_cand
            m += G[:, k] * inv_tau
            if max_events and len(idx_out) >= max_events:
                t = t_cand
                break
        else:
            m = m_cand
        t = t_cand

        if progress_every and len(idx_out) and len(idx_out) % progress_every == 0:
            print(f"  ... {len(idx_out)} events, t={t:.0f}/{sim_time_ms:.0f} ms, "
                  f"{n_candidates} candidates, {time.time()-t0:.1f}s elapsed", flush=True)

    idx = np.asarray(idx_out, dtype=np.int64)
    times_ms = np.asarray(times_out, dtype=np.float64)
    stats = dict(n_events=len(idx), n_candidates=n_candidates,
                 wall_s=time.time() - t0, mean_rate_hz=len(idx) / (N * sim_time_ms / 1000.0))
    return idx, times_ms, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adj", required=True, help="path to an existing adj_true.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau-syn", type=float, default=10.0, help="kernel decay (ms), Pernice 2011's own value")
    ap.add_argument("--rate0", type=float, default=10.0, help="baseline drive y0 (Hz), Pernice 2011's own value")
    ap.add_argument("--margin", type=float, default=0.3, help="stability margin: max|eig(G)| = 1-margin")
    ap.add_argument("--sim-time", type=float, default=100_000.0, help="ms, matches n1250_pif pilot")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-max", type=int, default=None, help="truncate network to first N neurons (smoke test)")
    ap.add_argument("--max-events", type=int, default=None, help="safety cap on total events")
    args = ap.parse_args()

    adj = np.load(Path(args.adj).expanduser()).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    if args.n_max:
        adj = adj[:args.n_max, :args.n_max].copy()
    N = adj.shape[0]
    types = np.sign(adj.sum(1)); types[types == 0] = 1
    nE, nI = int((types > 0).sum()), int((types < 0).sum())
    print(f"N={N} (E={nE}, I={nI})  tau_syn={args.tau_syn}ms  y0={args.rate0}Hz  "
          f"sim_time={args.sim_time}ms  seed={args.seed}", flush=True)

    G, ginfo = build_G(adj, args.margin)
    print(f"max|eig(G_raw)|={ginfo['rho_raw']:.4f}  coupling_scale={ginfo['coupling_scale']:.4g}  "
          f"max|eig(G_scaled)|={ginfo['rho_scaled']:.4f}  (must be < 1)", flush=True)

    print("simulating (exact thinning) ...", flush=True)
    idx, times_ms, stats = simulate_hawkes(
        G, args.rate0, args.tau_syn, args.sim_time, args.seed, max_events=args.max_events)
    print(f"done: {stats['n_events']} events ({stats['n_candidates']} candidates, "
          f"{stats['wall_s']:.1f}s wall)  mean rate={stats['mean_rate_hz']:.2f} Hz", flush=True)

    # --- closed-form Pernice sanity check (Eq. 10): y = (1-G)^-1 y0 -------- #
    B = np.linalg.inv(np.eye(N) - G)
    y0_vec = np.full(N, args.rate0)
    y_theory = B @ y0_vec
    rateE_theory, rateI_theory = y_theory[types > 0].mean(), y_theory[types < 0].mean()

    rate_sim = np.zeros(N)
    dur_s = args.sim_time / 1000.0
    counts = np.bincount(idx, minlength=N)
    rate_sim = counts / dur_s
    rateE_sim, rateI_sim = rate_sim[types > 0].mean(), rate_sim[types < 0].mean()
    print(f"rate check (Pernice Eq. 10 vs simulated, Fig. 2C-style):\n"
          f"  E: theory={rateE_theory:.2f} Hz  sim={rateE_sim:.2f} Hz\n"
          f"  I: theory={rateI_theory:.2f} Hz  sim={rateI_sim:.2f} Hz", flush=True)

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "idx.npy", idx.astype(np.int32))
    np.save(out / "times_ms.npy", times_ms.astype(np.float32))
    np.save(out / "adj_true.npy", adj)  # original, unscaled, [source,target]
    meta = dict(tau_syn=args.tau_syn, rate0=args.rate0, sim_time=args.sim_time,
                seed=args.seed, n_events=stats["n_events"], mean_rate_hz=stats["mean_rate_hz"],
                rateE_theory=float(rateE_theory), rateI_theory=float(rateI_theory),
                rateE_sim=float(rateE_sim), rateI_sim=float(rateI_sim),
                source_adj=str(args.adj), **ginfo)
    np.save(out / "meta.npy", meta, allow_pickle=True)
    print(f"wrote {out}/idx.npy times_ms.npy adj_true.npy meta.npy", flush=True)


if __name__ == "__main__":
    main()
