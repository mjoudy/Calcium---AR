"""
Dale-regularization candidates — head-to-head on the N=100 feed.

All methods share the base Elastic-Net loss
    (1/2T)||Y - A X||^2 + lam1||A||_1 + (lam2/2)||A||^2
and differ only in the Dale term. Candidates:

  C1  hard in-solver Dale  — fixed sign per column, sign projection each step (current champion)
  C2  soft Dale            — penalty lam_D * sum max(0, -t_j A_ij); sweep lam_D
  C3  iterative hard       — re-estimate signs from the cleaner fit, repeat (block-coordinate)
  C4  iterative soft (EM)  — per-neuron confidence scales lam_D; re-estimate each round
  C5  min() purity         — no sign guess; penalize the minority sign per column (non-convex)

References: OLS, EN(no Dale), and C1 with TRUE types (ceiling).

Usage:  python scripts/dale_candidates_test.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calcium_ar.solvers.fista import solve as fista_solve
from calcium_ar.experiments.metrics import connectivity_metrics
from calcium_ar.experiments.config import ExperimentConfig
from calcium_ar.experiments.result import ExperimentResult
from calcium_ar.experiments.ledger import append_row

DS = "results/solver_comparison_N100/dataset"
FEED = "results/regularization_test/feed.zarr"
OUT = "results/dale_candidates_test"
NE = 80
LAG = 15
LAM1, LAM2 = 3e-3, 1e-3
N_ITER = 500


# ---------------------------------------------------------------- helpers ---- #
def strongest_entry_types(A):
    """Sign of each neuron's strongest outgoing edge (column-wise)."""
    N = A.shape[0]; t = np.ones(N)
    for j in range(N):
        c = A[:, j].copy(); c[j] = 0.0
        t[j] = np.sign(c[np.argmax(np.abs(c))]) or 1.0
    return t


def column_masses(A):
    """Per-column positive mass and absolute negative mass (diagonal excluded)."""
    B = A.copy(); np.fill_diagonal(B, 0.0)
    pos = np.maximum(B, 0.0).sum(axis=0)
    neg = np.maximum(-B, 0.0).sum(axis=0)
    return pos, neg


def mass_confidence(A):
    """Confidence in each neuron's type: |pos-neg| / (pos+neg), in [0,1]."""
    pos, neg = column_masses(A)
    return np.abs(pos - neg) / (pos + neg + 1e-12)


def soft_threshold(U, thr):
    return np.sign(U) * np.maximum(np.abs(U) - thr, 0.0)


def soft_dale_prox(U, types, thr_correct, thr_wrong):
    """Asymmetric soft-threshold: cheap on the correct sign, expensive on the wrong one.
    thr_correct / thr_wrong may be scalars or (1, N) arrays (per-column)."""
    tcol = types[None, :]
    thr_pos = np.where(tcol > 0, thr_correct, thr_wrong)   # threshold for positive entries
    thr_neg = np.where(tcol > 0, thr_wrong, thr_correct)   # threshold for negative entries
    return np.maximum(U - thr_pos, 0.0) + np.minimum(U + thr_neg, 0.0)


def fista_core(Cxx, Cyx, lam2, n_iter, prox):
    """FISTA driver. `prox(U, step, A_prev, k) -> V` applies the non-smooth part."""
    N = Cxx.shape[0]
    L = float(np.linalg.eigvalsh(Cxx)[-1]) + lam2
    step = 1.0 / L
    A = np.zeros((N, N)); Z = A.copy(); tk = 1.0
    for k in range(n_iter):
        grad = Z @ Cxx - Cyx + lam2 * Z
        U = Z - step * grad
        V = prox(U, step, A, k)
        tnew = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
        Z = V + ((tk - 1.0) / tnew) * (V - A)
        A, tk = V, tnew
    np.fill_diagonal(A, 0.0)
    return A


# ---------------------------------------------------------- the candidates ---- #
def c1_hard(Cxx, Cyx, types, lam1=LAM1, lam2=LAM2, n_iter=N_ITER):
    def prox(U, s, A, k):
        V = soft_threshold(U, s * lam1)
        return types[None, :] * np.maximum(types[None, :] * V, 0.0)   # sign projection
    return fista_core(Cxx, Cyx, lam2, n_iter, prox)


def c2_soft(Cxx, Cyx, types, lam_dale, lam1=LAM1, lam2=LAM2, n_iter=N_ITER):
    def prox(U, s, A, k):
        return soft_dale_prox(U, types, s * lam1, s * (lam1 + lam_dale))
    return fista_core(Cxx, Cyx, lam2, n_iter, prox)


def c3_iter_hard(Cxx, Cyx, types0, lam1=LAM1, lam2=LAM2, n_iter=300, rounds=6):
    t = types0.copy()
    for _ in range(rounds):
        A = c1_hard(Cxx, Cyx, t, lam1, lam2, n_iter)
        t_new = strongest_entry_types(A)
        if np.array_equal(t_new, t):
            break
        t = t_new
    return A, t


def c4_iter_soft(Cxx, Cyx, A_init, lam_dale_base, lam1=LAM1, lam2=LAM2, n_iter=300, rounds=6):
    t = strongest_entry_types(A_init)
    c = mass_confidence(A_init)
    A = A_init
    for _ in range(rounds):
        lam_col = (lam_dale_base * c)[None, :]            # weak where unsure, strong where confident
        def prox(U, s, A_prev, k, _t=t, _lc=lam_col):
            return soft_dale_prox(U, _t, s * lam1, s * (lam1 + _lc))
        A = fista_core(Cxx, Cyx, lam2, n_iter, prox)
        t_new, c = strongest_entry_types(A), mass_confidence(A)
        if np.array_equal(t_new, t):
            break
        t = t_new
    return A, t


def c5_min_purity(Cxx, Cyx, lam_dale, lam1=LAM1, lam2=LAM2, n_iter=600, warm=150):
    def prox(U, s, A, k):
        if k < warm:                                       # warm-start: plain EN
            return soft_threshold(U, s * lam1)
        pos, neg = column_masses(A)
        t_eff = np.where(pos >= neg, 1.0, -1.0)            # keep the majority sign
        return soft_dale_prox(U, t_eff, s * lam1, s * (lam1 + lam_dale))
    return fista_core(Cxx, Cyx, lam2, n_iter, prox)


# ------------------------------------------------------------------- main ---- #
def main():
    adj = np.load(Path(DS) / "adj_true.npy"); np.fill_diagonal(adj, 0.0)
    feed = np.asarray(zarr.open(FEED, "r")[:])
    N = adj.shape[0]; off = ~np.eye(N, dtype=bool)
    m = connectivity_metrics._metrics

    X = feed[:, :-LAG] - feed[:, :-LAG].mean(1, keepdims=True)
    Y = feed[:, LAG:] - feed[:, LAG:].mean(1, keepdims=True)
    M = X.shape[1]
    Cxx = (X @ X.T) / M
    Cyx = (Y @ X.T) / M

    def ei(A):
        g = adj.T[off]; a = np.abs(A[off]); inh, exc = a[g < 0], a[g > 0]
        return float(np.median(inh) / np.median(exc)) if len(inh) and len(exc) and np.median(exc) else float("nan")

    def score(A):
        return dict(spearman=m["spearman"](A, adj), pearson=m["pearson"](A, adj),
                    auc=m["auc_roc"](A, adj), f1=m["f1"](A, adj),
                    precision=m["precision"](A, adj), type_acc=m["dale_type_accuracy"](A, adj),
                    ei=ei(A), dale=m["degree_of_daleianity"](A, adj))

    # ---- baselines / references ----
    A_ols = Cyx @ np.linalg.inv(Cxx + 1e-9 * np.eye(N)); np.fill_diagonal(A_ols, 0.0)
    A_en = fista_solve(FEED, lag=LAG, lam_l1=LAM1, lam_l2=LAM2, n_iter=N_ITER, chunk_size=10000)
    np.fill_diagonal(A_en, 0.0)
    t_unsup = strongest_entry_types(A_en)
    t_true = np.where(np.arange(N) < NE, 1.0, -1.0)

    methods = [
        ("OLS",                       "baseline", False, A_ols),
        ("EN(no Dale)",               "regularize", False, A_en),
        ("C1_hard",                   "dale-hard", False, c1_hard(Cxx, Cyx, t_unsup)),
        ("C1_hard(true types)",       "dale-hard*", True, c1_hard(Cxx, Cyx, t_true)),
    ]
    # C2 — soft Dale sweep
    for ld in (3e-3, 1e-2, 3e-2, 1e-1):
        methods.append((f"C2_soft(lD={ld:g})", "dale-soft", False, c2_soft(Cxx, Cyx, t_unsup, ld)))
    # C3 — iterative hard
    A_c3, _ = c3_iter_hard(Cxx, Cyx, t_unsup)
    methods.append(("C3_iter_hard", "dale-iter", False, A_c3))
    # C4 — iterative soft (EM)
    A_c4, _ = c4_iter_soft(Cxx, Cyx, A_en, lam_dale_base=3e-2)
    methods.append(("C4_iter_soft_EM", "dale-em", False, A_c4))
    # C5 — min() purity (no guess)
    for ld in (3e-3, 1e-2, 3e-2):
        methods.append((f"C5_min(lD={ld:g})", "dale-min", False, c5_min_purity(Cxx, Cyx, ld)))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    out_dir = Path(OUT); out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["spearman", "pearson", "auc", "f1", "precision", "type_acc", "ei", "dale"]
    print(f"{'method':>22}{'GT':>4}" + "".join(f"{c[:8]:>9}" for c in cols))
    print("-" * (26 + 9 * len(cols)))
    for name, kind, gt, A in methods:
        s = score(A)
        print(f"{name:>22}{('Y' if gt else '·'):>4}" + "".join(f"{s[c]:>9.3f}" for c in cols))
        cfg = ExperimentConfig(name=f"dalecand/{name}", lag_ms=LAG * 0.1, dt=0.1,
                               n_excitatory=NE, n_inhibitory=N - NE, solver="fista",
                               output_dir=OUT, data_path=DS)
        res = ExperimentResult(config_path="-", loss_curve=[], duration_seconds=0.0, timestamp=ts,
                               run_dir=str(out_dir / name.replace("(", "_").replace(")", "").replace("=", "").replace(" ", "_")),
                               spikes_path=None, calcium_path=None, feed_zarr_path=FEED,
                               adj_true_path=str(Path(DS) / "adj_true.npy"), adj_inferred_path="-",
                               metrics={f"connectivity/{k}": s[k] for k in ("spearman", "pearson", "auc", "f1", "precision")}
                                       | {"diag/ei_ratio": s["ei"], "diag/daleianity": s["dale"], "diag/type_acc": s["type_acc"]})
        append_row(res, cfg, out_dir / "ledger.csv")


if __name__ == "__main__":
    main()
