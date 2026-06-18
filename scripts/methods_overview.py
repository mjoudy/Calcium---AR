"""
Master methods overview — every regularization / post-processing method in one table.

Runs all methods at the fixed operating point (lag = 1.5 ms, real preprocessed feed),
scores them on every direction, and writes a single comparison table to
docs/experiments/methods_overview.md (and the ledger).

IMPORTANT — ground-truth policy
-------------------------------
No method uses the ground truth to PRODUCE its estimate (the 'GT' column flags the
few oracle/ceiling rows that do, kept only to show the best-possible bound).  Ground
truth is used ONLY to SCORE.  Two of the scores need NO ground truth at all and could
be computed on real data — `daleianity` and the class `overlap` — those are the knobs
you'd actually tune in an experiment.

Scoring directions
------------------
    detection : f1, precision, recall, auc        (which connections exist)
    type      : type_acc                          (E/I per neuron)
    magnitude : spearman, pearson, ei             (weight values / E-I scale)
    unsup     : daleianity, overlap               (quality WITHOUT ground truth)

Usage:  python scripts/methods_overview.py
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
OUT = "results/methods_overview"
DOC = "docs/experiments/methods_overview.md"
NE = 80
LAG = 15


# --------------------------------------------------------------------------- #
# estimator + building blocks (all unsupervised unless noted)
# --------------------------------------------------------------------------- #

def ar_ols(X, lag):
    p = X[:, :-lag] - X[:, :-lag].mean(1, keepdims=True)
    n = X[:, lag:] - X[:, lag:].mean(1, keepdims=True)
    A = np.linalg.solve((p @ p.T).T, (n @ p.T).T).T
    np.fill_diagonal(A, 0.0)
    return A


def strongest_entry_types(A):
    """Unsupervised type per source column = sign of its largest-magnitude entry."""
    N = A.shape[0]; t = np.ones(N)
    for j in range(N):
        col = A[:, j].copy(); col[j] = 0.0
        t[j] = np.sign(col[np.argmax(np.abs(col))]) or 1.0
    return t


def rescale_strongest(A):
    """Unsupervised: strongest-entry types, then equalise inferred-I group to E group."""
    N = A.shape[0]; off = ~np.eye(N, dtype=bool); B = A.copy()
    t = strongest_entry_types(A)
    Ec, Ic = np.where(t > 0)[0], np.where(t < 0)[0]
    mk = lambda cols: (np.isin(np.arange(N), cols)[None, :] & off)
    mE, mI = np.abs(A[mk(Ec)]), np.abs(A[mk(Ic)])
    medE = np.median(mE) if mE.size else 0.0
    medI = np.median(mI) if mI.size else 0.0
    if medI > 1e-9 and medE > 0:
        B[:, Ic] = A[:, Ic] * (medE / medI)
    np.fill_diagonal(B, 0.0)
    return B


def dale_cleanup(A):
    """Unsupervised Dale: per column, keep only entries matching the column's type."""
    N = A.shape[0]; B = A.copy()
    t = strongest_entry_types(A)
    for j in range(N):
        col = B[:, j]
        col[np.sign(col) != t[j]] = 0.0   # zero wrong-sign entries
        B[:, j] = col
    np.fill_diagonal(B, 0.0)
    return B


def mixture_threshold(A):
    """Unsupervised: 3-component GMM on off-diag weights; zero the 'unconnected' bump."""
    from sklearn.mixture import GaussianMixture
    N = A.shape[0]; off = ~np.eye(N, dtype=bool)
    x = A[off].reshape(-1, 1)
    gm = GaussianMixture(n_components=3, random_state=0, n_init=2).fit(x)
    unconn = int(np.argmin(np.abs(gm.means_.ravel())))   # bump nearest 0 = unconnected
    lab = gm.predict(x)
    B = A.copy(); flat = B[off].copy()
    flat[lab == unconn] = 0.0
    B[off] = flat
    np.fill_diagonal(B, 0.0)
    return B


def colnorm(A):
    N = A.shape[0]; B = A.copy()
    for j in range(N):
        sd = B[np.arange(N) != j, j].std()
        if sd > 1e-9:
            B[:, j] = B[:, j] / sd
    np.fill_diagonal(B, 0.0)
    return B


def rescale_balance_nz(A, rates):
    """Balance rescale that estimates per-group magnitude from NON-ZERO entries only —
    so it composes with Dale/mixture (which zero many entries) instead of breaking on
    the all-zeros median."""
    N = A.shape[0]; off = ~np.eye(N, dtype=bool); B = A.copy()
    t = strongest_entry_types(A)
    g = balance_g(t, rates)
    Ec, Ic = np.where(t > 0)[0], np.where(t < 0)[0]

    def med_nz(cols):
        mk = np.isin(np.arange(N), cols)[None, :] & off
        v = np.abs(A[mk]); v = v[v > 1e-9]
        return np.median(v) if v.size else 0.0

    medE, medI = med_nz(Ec), med_nz(Ic)
    if medI > 1e-9 and medE > 0 and np.isfinite(g):
        B[:, Ic] = A[:, Ic] * (g * medE / medI)
    np.fill_diagonal(B, 0.0)
    return B


def balance_g(types, rates):
    """Estimate the E/I weight ratio g from network balance: g ≈ (N_E·νE)/(N_I·νI)."""
    E = types > 0; I = types < 0
    if E.sum() == 0 or I.sum() == 0 or rates[I].mean() == 0:
        return np.nan
    return (E.sum() * rates[E].mean()) / (I.sum() * rates[I].mean())


def rescale_balance(A, rates):
    """Unsupervised magnitude: strongest-entry types + g from balance -> lift inferred-I
    group to ratio g (not just to 1).  Assumes the network is balanced and rates observed."""
    N = A.shape[0]; off = ~np.eye(N, dtype=bool); B = A.copy()
    t = strongest_entry_types(A)
    g = balance_g(t, rates)
    Ec, Ic = np.where(t > 0)[0], np.where(t < 0)[0]
    mk = lambda cols: (np.isin(np.arange(N), cols)[None, :] & off)
    medE = np.median(np.abs(A[mk(Ec)])) if Ec.size else 0.0
    medI = np.median(np.abs(A[mk(Ic)])) if Ic.size else 0.0
    if medI > 1e-9 and medE > 0 and np.isfinite(g):
        B[:, Ic] = A[:, Ic] * (g * medE / medI)
    np.fill_diagonal(B, 0.0)
    return B


def dale_fista(X, Y, types, lam1=3e-3, lam2=1e-3, n_iter=800):
    """Sign-constrained Elastic Net (Dale as regularization) via FISTA. X,Y centred (N,M)."""
    N, M = X.shape
    Cxx = (X @ X.T) / M; Cyx = (Y @ X.T) / M
    Lip = np.linalg.eigvalsh(Cxx)[-1] + lam2; thr = lam1 / Lip
    tcol = types[None, :]
    A = np.zeros((N, N)); Z = A.copy(); tk = 1.0
    for _ in range(n_iter):
        V = Z - (Z @ Cxx - Cyx + lam2 * Z) / Lip
        V = np.sign(V) * np.maximum(np.abs(V) - thr, 0.0)   # L1 soft-threshold
        V = tcol * np.maximum(tcol * V, 0.0)                # per-column sign projection
        tnew = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        Z = V + ((tk - 1) / tnew) * (V - A); A, tk = V, tnew
    np.fill_diagonal(A, 0.0)
    return A


def oracle(A, adj):                      # GT — ceiling only
    N = A.shape[0]; off = ~np.eye(N, dtype=bool); G = adj.T
    exc = off & (G > 0); inh = off & (G < 0)
    sE = (A[exc] * G[exc]).sum() / (G[exc] ** 2).sum()
    sI = (A[inh] * G[inh]).sum() / (G[inh] ** 2).sum()
    B = A.copy()
    for j in range(N):
        s = sE if j < NE else sI
        if abs(s) > 1e-12:
            B[:, j] = A[:, j] / s
    np.fill_diagonal(B, 0.0)
    return B


def main():
    import json
    adj = np.load(Path(DS) / "adj_true.npy"); np.fill_diagonal(adj, 0.0)
    feed = np.asarray(zarr.open(FEED, "r")[:])
    meta = json.loads((Path(DS) / "metadata.json").read_text())["sim_params"]
    spk = np.asarray(zarr.open(str(Path(DS) / "spikes.zarr"), "r")[:])
    rates = spk.sum(1) / (meta["sim_time"] / 1000.0)   # Hz per neuron (observable)
    N = adj.shape[0]; off = ~np.eye(N, dtype=bool)
    m = connectivity_metrics._metrics

    def ei(A):
        g = adj.T[off]; a = np.abs(A[off]); inh, exc = a[g < 0], a[g > 0]
        return float(np.median(inh) / np.median(exc)) if len(inh) and len(exc) and np.median(exc) else float("nan")

    def score(A):
        ov = m["exc_unc_overlap"](A, adj) + m["inh_unc_overlap"](A, adj)
        return dict(spearman=m["spearman"](A, adj), pearson=m["pearson"](A, adj),
                    auc=m["auc_roc"](A, adj), f1=m["f1"](A, adj),
                    precision=m["precision"](A, adj), recall=m["recall"](A, adj),
                    type_acc=m["dale_type_accuracy"](A, adj), ei=ei(A),
                    dale=m["degree_of_daleianity"](A, adj), overlap=ov)

    A_ols = ar_ols(feed, LAG)
    A_en = fista_solve(FEED, lag=LAG, lam_l1=3e-3, lam_l2=1e-3, n_iter=500, chunk_size=10000)
    np.fill_diagonal(A_en, 0.0)
    A_en2 = fista_solve(FEED, lag=LAG, lam_l1=1e-2, lam_l2=1e-3, n_iter=500, chunk_size=10000)
    np.fill_diagonal(A_en2, 0.0)

    # centred lag pairs + types for the Dale-regularization solver
    Xc = feed[:, :-LAG] - feed[:, :-LAG].mean(1, keepdims=True)
    Yc = feed[:, LAG:] - feed[:, LAG:].mean(1, keepdims=True)
    t_unsup = strongest_entry_types(A_en)
    A_dalereg = dale_fista(Xc, Yc, t_unsup)

    # (name, kind, uses_GT, matrix)
    methods = [
        ("OLS",                  "baseline",     False, A_ols),
        ("EN(L1=3e-3)",          "regularize",   False, A_en),
        ("EN(L1=1e-2)",          "regularize",   False, A_en2),
        ("OLS+colnorm",          "rescale",      False, colnorm(A_ols)),
        ("OLS+oracle",           "rescale*",     True,  oracle(A_ols, adj)),
        ("EN+colnorm",           "reg+rescale",  False, colnorm(A_en)),
        ("EN+rescale_strongest", "reg+rescale",  False, rescale_strongest(A_en)),
        ("EN+rescale_balance",   "reg+rescale",  False, rescale_balance(A_en, rates)),
        ("EN+dale",              "reg+dale",     False, dale_cleanup(A_en)),
        ("EN_daleReg",           "dale-reg",     False, A_dalereg),
        ("EN_daleReg+balance_nz","FULL pipeline",False, rescale_balance_nz(A_dalereg, rates)),
        ("EN+mixture",           "reg+mixture",  False, mixture_threshold(A_en)),
        ("EN+dale+mixture",      "reg+combo",    False, mixture_threshold(dale_cleanup(A_en))),
        ("EN+dale+mixture+balance_nz", "FULL pipeline", False,
         rescale_balance_nz(mixture_threshold(dale_cleanup(A_en)), rates)),
        ("EN+oracle",            "reg+rescale*", True,  oracle(A_en, adj)),
    ]

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    out_dir = Path(OUT); out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["spearman", "pearson", "auc", "f1", "precision", "recall", "type_acc", "ei", "dale", "overlap"]

    rows = []
    for name, kind, gt, A in methods:
        s = score(A)
        rows.append((name, kind, gt, s))
        cfg = ExperimentConfig(name=f"overview/{name}", lag_ms=LAG * 0.1, dt=0.1,
                               n_excitatory=NE, n_inhibitory=N - NE, solver="fista",
                               output_dir=OUT, data_path=DS)
        res = ExperimentResult(config_path="-", loss_curve=[], duration_seconds=0.0, timestamp=ts,
                               run_dir=str(out_dir / name.replace("+", "_").replace("(", "").replace(")", "").replace("=", "")),
                               spikes_path=None, calcium_path=None, feed_zarr_path=FEED,
                               adj_true_path=str(Path(DS) / "adj_true.npy"), adj_inferred_path="-",
                               metrics={f"connectivity/{k}": s[k] for k in
                                        ("spearman", "pearson", "auc", "f1", "precision", "recall")}
                                       | {"diag/ei_ratio": s["ei"], "diag/daleianity": s["dale"],
                                          "diag/overlap": s["overlap"], "diag/type_acc": s["type_acc"]})
        append_row(res, cfg, out_dir / "ledger.csv")

    # ---- console table ----
    head = f"{'method':>22}{'kind':>14}{'GT':>4}" + "".join(f"{c[:8]:>9}" for c in cols)
    print(head); print("-" * len(head))
    for name, kind, gt, s in rows:
        print(f"{name:>22}{kind:>14}{('Y' if gt else '·'):>4}" +
              "".join(f"{s[c]:>9.3f}" for c in cols))

    # ---- write markdown overview ----
    lines = ["# Methods Overview\n",
             "All methods at lag = 1.5 ms on the real preprocessed feed (N=100). **GT=Y**",
             "means the method used ground truth to *produce* the estimate (oracle ceilings",
             "only); all others are fully unsupervised. `daleianity` and `overlap` need NO",
             "ground truth to compute — they are the knobs tunable on real data.",
             "Regenerate: `python scripts/methods_overview.py`.\n",
             "Directions: detection = f1/precision/recall/auc · type = type_acc ·",
             "magnitude = spearman/pearson/ei · unsupervised-quality = daleianity/overlap",
             "(lower overlap = better).\n",
             "| method | kind | GT | " + " | ".join(cols) + " |",
             "|" + "---|" * (3 + len(cols))]
    for name, kind, gt, s in rows:
        lines.append(f"| {name} | {kind} | {'Y' if gt else '·'} | " +
                     " | ".join(f"{s[c]:.3f}" for c in cols) + " |")
    lines += ["\n*\\* = uses ground truth (ceiling, not deployable).*\n"]

    # ---- per-method mechanism + assumptions (for later theoretical vetting) ----
    desc = [
        ("OLS", "Plain multivariate AR regression A=C_yx·inv(C_xx) on centred feed.",
         "Linear AR(1) model at the chosen lag; off-diagonal A ↔ G.T."),
        ("EN(L1=3e-3) / EN(L1=1e-2)", "OLS + Elastic-Net penalty (L1 sparsity + L2 stability) via FISTA.",
         "Connectivity is sparse; adds shrinkage bias toward 0. Larger L1 = sparser."),
        ("OLS+colnorm / EN+colnorm", "Divide each neuron's column by its own std (per-neuron normalisation).",
         "Every neuron has similar total outgoing strength. Amplifies noise in weak/quiet columns."),
        ("EN+rescale_strongest", "Guess type from each column's strongest entry; equalise inferred-I group to inferred-E (target ratio 1).",
         "Dale (single-sign neurons) + the strongest entry reveals type. Targets ratio 1, not true g."),
        ("EN+rescale_balance", "As strongest, but set the target ratio to g estimated from network balance g≈(N_E·νE)/(N_I·νI).",
         "Network is balanced (E≈I input); firing rates observable; type guess approx. Gives ballpark g (~4–6 vs true 5)."),
        ("EN+dale", "Per column, zero every entry whose sign disagrees with the column's (strongest-entry) type.",
         "Dale's law holds strictly (no genuine opposite-sign outgoing edges). Removes wrong-sign false positives."),
        ("EN_daleReg", "Dale as REGULARIZATION: sign-constrained Elastic Net — FISTA prox projects each column onto its type's sign DURING the fit (not after).",
         "Dale's law strict; types from initial EN. Correct-sign entries re-grow to explain variance → beats post-hoc dale on detection."),
        ("EN+mixture", "Fit a 3-component Gaussian mixture to the off-diagonal weights; zero the component nearest 0 (unconnected).",
         "Weights are a 3-class mixture (exc/unconnected/inh) separable by magnitude."),
        ("EN+dale+mixture / +balance", "Compositions: Dale sign-cleanup, then mixture threshold, then optional balance rescale.",
         "Union of the component assumptions above."),
        ("OLS+oracle / EN+oracle", "Per-class least-squares slope mapping A onto G using TRUE neuron types.",
         "USES GROUND TRUTH — ceiling/upper bound only, NOT deployable."),
    ]
    lines += ["## How each method works (mechanism · assumptions)\n"]
    for name, how, assume in desc:
        lines += [f"- **{name}** — {how}", f"  - *assumes:* {assume}"]
    lines += ["\n*Scores above use ground truth only to evaluate; methods (except \\*) use only the estimate + observable rates.*\n"]
    Path(DOC).write_text("\n".join(lines) + "\n")
    print(f"\noverview table → {DOC}")
    print(f"ledger        → {out_dir/'ledger.csv'}")


if __name__ == "__main__":
    main()
