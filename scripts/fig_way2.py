"""
Way 2 — the correlational test: fake strength vs HIDDEN vs OBSERVED shared input.

If the false positives come from HIDDEN common input (not from shared input among
the recorded neurons, which OLS already removes), then a false-positive edge's
magnitude should:
  - RISE with the number of HIDDEN common presynaptic drivers, and
  - be FLAT vs the number of OBSERVED common drivers (OLS conditioned those away).

Observe a subset S (default 50%). For each false-positive pair (i,j) in S, count
their common presynaptic neurons split into observed (in S) and hidden (not in S),
and bin the inferred |weight| against each. Runs off cached moments — local,
seconds, no GPU.

Usage:
  python scripts/fig_way2.py --data ~/calcium_results/best_moments/n1250r4 \
      --out figures/fig_way2_n1250
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs


def ols_block(Cxx, Cyx, S):
    ix = np.ix_(S, S)
    A = Cyx[ix] @ np.linalg.inv(Cxx[ix] + 1e-9 * np.eye(len(S)))
    np.fill_diagonal(A, 0.0)
    return A


def binned(count, w, min_n):
    """mean |w| per integer count value, keeping bins with >= min_n pairs."""
    vals = np.arange(count.min(), count.max() + 1)
    xs, ys, ns = [], [], []
    for v in vals:
        m = count == v
        if m.sum() >= min_n:
            xs.append(v); ys.append(w[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--obs-frac", type=float, default=0.5)
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--min-n", type=int, default=300, help="min pairs per bin")
    ap.add_argument("--max-fp", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figures/fig_way2")
    args = ap.parse_args()

    d = Path(args.data)
    Cxx = np.load(d / "Cxx.npy"); Cyx = np.load(d / "Cyx.npy")
    adj = np.load(d / "adj_true.npy").astype(np.float64)
    N = Cxx.shape[0]; np.fill_diagonal(adj, 0.0)
    rng = np.random.default_rng(args.seed)

    types = np.sign(adj.sum(1)); types[types == 0] = 1
    E = np.flatnonzero(types > 0); I = np.flatnonzero(types < 0)
    rng.shuffle(E); rng.shuffle(I)
    nE, nI = int(args.obs_frac * len(E)), int(args.obs_frac * len(I))
    S = np.sort(np.concatenate([E[:nE], I[:nI]]))
    hidden = np.setdiff1d(np.arange(N), S)
    Sset = np.zeros(N, bool); Sset[S] = True
    print(f"N={N}  observed |S|={len(S)}  hidden={len(hidden)}")

    A = ols_block(Cxx, Cyx, S)
    aa = np.abs(A)
    tau = np.quantile(aa[~np.eye(len(S), dtype=bool)], 1.0 - args.density)
    gb = adj[np.ix_(S, S)].T                          # true in A orientation
    fp = (aa > tau) & (gb == 0); np.fill_diagonal(fp, False)
    fa, fb = np.nonzero(fp)
    if len(fa) > args.max_fp:
        k = rng.choice(len(fa), args.max_fp, replace=False); fa, fb = fa[k], fb[k]
    ti, tj = S[fa], S[fb]; w = aa[fa, fb]
    print(f"false positives: {len(fa)}")

    # common presynaptic drivers (sources into both ti and tj), split obs/hidden.
    # Bc[k, t] = 1 if k -> t exists (adj[source, target]).
    Bc = (adj != 0)
    both = Bc[:, ti] & Bc[:, tj]                       # (N, n_fp): k drives both
    obs_common = (both & Sset[:, None]).sum(0)         # observed common drivers
    hid_common = (both & (~Sset)[:, None]).sum(0)      # hidden common drivers

    xo, yo, no = binned(obs_common, w, args.min_n)
    xh, yh, nh = binned(hid_common, w, args.min_n)

    fs.apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.6))
    fig.subplots_adjust(left=0.11, right=0.96, top=0.88, bottom=0.13)
    ax.plot(xh, yh, "o-", color="#c0392b", lw=2, ms=6, label="HIDDEN common drivers")
    ax.plot(xo, yo, "o-", color="#2a78d6", lw=2, ms=6, label="OBSERVED common drivers")
    ax.set(xlabel="number of common presynaptic drivers of the pair",
           ylabel="mean |inferred weight| of false-positive edges")
    ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_title(f"N={N}, {int(args.obs_frac*100)}% observed: fakes grow with "
                 "HIDDEN shared input, flat vs OBSERVED",
                 fontsize=12, color=fs.INK)
    fs.save(fig, args.out)
    # slopes for the record
    def slope(x, y):
        return np.polyfit(x, y, 1)[0] if len(x) > 2 else float("nan")
    print(f"slope vs hidden={slope(xh, yh):.3e}   vs observed={slope(xo, yo):.3e}")


if __name__ == "__main__":
    main()
