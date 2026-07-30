"""
Way 3 — the causal "add the hidden driver back" test (local, N=1250).

Claim to prove: the false-positive (symmetric) edges are caused by HIDDEN common
input, not by shared input among the recorded neurons (OLS already removes that
via its inverse / partial-correlation step).

Test:
  1. Observe a subset S (default 50%). Solve OLS on the sub-block -> false-positive
     edges (i,j) in S: predicted (top-density |A|) but NOT truly connected.
  2. Find the HIDDEN common drivers of those pairs (neurons in the unobserved set
     that project to BOTH i and j).
  3. Add those drivers back to the observed set, re-solve -> the fakes should
     COLLAPSE (now OLS can condition on the culprit).
  4. CONTROL: add the same number of RANDOM hidden neurons -> the fakes should NOT
     collapse.

If (3) shrinks the fakes and (4) does not, hidden common input is the cause.

Runs off the cached moments (Cxx, Cyx) + adj_true — no simulation, no GPU.

Usage:
  python scripts/fig_way3.py --data ~/calcium_results/best_moments/n1250r4 \
      --out figures/fig_way3_n1250
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with Cxx.npy, Cyx.npy, adj_true.npy")
    ap.add_argument("--obs-frac", type=float, default=0.5)
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--n-add", type=int, default=150, help="hidden neurons to add back")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figures/fig_way3")
    args = ap.parse_args()

    d = Path(args.data)
    Cxx = np.load(d / "Cxx.npy"); Cyx = np.load(d / "Cyx.npy")
    adj = np.load(d / "adj_true.npy").astype(np.float64)
    N = Cxx.shape[0]; np.fill_diagonal(adj, 0.0)
    rng = np.random.default_rng(args.seed)

    # observed subset S (stratified by E/I), hidden = complement
    types = np.sign(adj.sum(1)); types[types == 0] = 1
    E = np.flatnonzero(types > 0); I = np.flatnonzero(types < 0)
    rng.shuffle(E); rng.shuffle(I)
    nE, nI = int(args.obs_frac * len(E)), int(args.obs_frac * len(I))
    S = np.sort(np.concatenate([E[:nE], I[:nI]]))
    hidden = np.setdiff1d(np.arange(N), S)
    print(f"N={N}  observed |S|={len(S)}  hidden={len(hidden)}")

    A = ols_block(Cxx, Cyx, S)                       # A[a,b] = edge S[b]->S[a]
    aa = np.abs(A)
    tau = np.quantile(aa[~np.eye(len(S), dtype=bool)], 1.0 - args.density)
    # false positives: predicted (|A|>tau) but no true edge S[b]->S[a]
    gb = adj[np.ix_(S, S)].T                         # true in A orientation: adj[S[b],S[a]]
    fp = (aa > tau) & (gb == 0)
    np.fill_diagonal(fp, False)
    fa, fb = np.nonzero(fp)                          # positions in S
    print(f"false-positive edges at {int(args.density*100)}%: {len(fa)}")

    # hidden common drivers of the fp target pairs (i=S[fa], j=S[fb]):
    # neurons k in hidden with adj[k, i] != 0 AND adj[k, j] != 0
    Bh = (adj[hidden][:, :] != 0)                    # (n_hidden, N): k -> target
    ti, tj = S[fa], S[fb]
    # count, per hidden neuron, how many fp pairs it drives in common
    drives = Bh[:, ti] & Bh[:, tj]                   # (n_hidden, n_fp)
    score = drives.sum(1)
    top = hidden[np.argsort(-score)[:args.n_add]]    # the culprit hidden drivers
    randh = rng.choice(hidden, min(args.n_add, len(hidden)), replace=False)
    print(f"adding {len(top)} hidden drivers (top by shared-fp count) vs "
          f"{len(randh)} random hidden")

    def fp_weights(extra):
        Sp = np.sort(np.concatenate([S, extra]))
        Ap = ols_block(Cxx, Cyx, Sp)
        posp = {g: k for k, g in enumerate(Sp)}      # global -> position in Sp
        w = np.array([abs(Ap[posp[i], posp[j]]) for i, j in zip(ti, tj)])
        return w

    w_base = aa[fa, fb]
    w_drivers = fp_weights(top)
    w_random = fp_weights(randh)
    print(f"mean |fake weight|  baseline={w_base.mean():.4e}  "
          f"+drivers={w_drivers.mean():.4e}  +random={w_random.mean():.4e}")
    print(f"median reduction    +drivers={1-np.median(w_drivers)/np.median(w_base):.1%}  "
          f"+random={1-np.median(w_random)/np.median(w_base):.1%}")

    # ---- plot ----
    fs.apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.13, wspace=0.24)

    labels = ["baseline\n(50% observed)", f"+{len(top)} hidden\nDRIVERS",
              f"+{len(randh)} hidden\nRANDOM"]
    med = [np.median(w_base), np.median(w_drivers), np.median(w_random)]
    cols = [fs.C_NONE, "#3a9b52", "#c0392b"]
    ax[0].bar(range(3), med, color=cols, width=0.6)
    for x, m in enumerate(med):
        ax[0].text(x, m, f"{m:.2e}", ha="center", va="bottom", fontsize=9)
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(labels, fontsize=9)
    ax[0].set(ylabel="median |inferred weight| of the false-positive edges")
    ax[0].grid(True, axis="y", color=fs.GRID, lw=0.6); ax[0].set_axisbelow(True)
    fs.despine(ax[0]); ax[0].set_title("adding the hidden driver collapses the fakes")

    # per-edge: baseline vs +drivers
    ax[1].scatter(w_base, w_drivers, s=8, alpha=0.3, color="#3a9b52", label="+ hidden drivers")
    ax[1].scatter(w_base, w_random, s=8, alpha=0.3, color="#c0392b", label="+ random hidden")
    hi = max(w_base.max(), 1e-9)
    ax[1].plot([0, hi], [0, hi], color=fs.INK, ls=":", lw=1, label="no change")
    ax[1].set(xlabel="|weight| baseline (50% observed)",
              ylabel="|weight| after adding neurons", xlim=(0, hi), ylim=(0, hi))
    ax[1].grid(True, color=fs.GRID, lw=0.6); ax[1].set_axisbelow(True); fs.despine(ax[1])
    ax[1].legend(fontsize=9, loc="upper left", markerscale=2)
    ax[1].set_title("below the line = the fake shrank")

    fig.suptitle(f"Hidden neurons cause the fakes (N={N}): re-observing the shared "
                 "driver removes the false edge, a random neuron does not",
                 fontsize=12.5, color=fs.INK, x=0.07, ha="left", y=0.965)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
