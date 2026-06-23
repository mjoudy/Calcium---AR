"""
Metrics report — score every solution on the full metric set, including the new
3-class macro precision/recall/F1 and the per-class INHIBITORY numbers (the weak spot).

Solutions: OLS -> EN -> EN+Dale -> EN+Dale+balance -> EN+dale+mixture+balance.

Usage:  python scripts/metrics_report.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from calcium_ar.solvers.fista import solve as fista_solve
from calcium_ar.experiments.metrics import connectivity_metrics
from calcium_ar.experiments import metrics as M

DS = "results/solver_comparison_N100/dataset"
FEED = "results/regularization_test/feed.zarr"
NE = 80
LAG = 15
LAM1, LAM2 = 3e-3, 1e-3


def strongest_entry_types(A):
    N = A.shape[0]; t = np.ones(N)
    for j in range(N):
        c = A[:, j].copy(); c[j] = 0.0
        t[j] = np.sign(c[np.argmax(np.abs(c))]) or 1.0
    return t


def dale_fista(X, Y, types, lam1=LAM1, lam2=LAM2, n_iter=500):
    N, Mm = X.shape
    Cxx = (X @ X.T) / Mm; Cyx = (Y @ X.T) / Mm
    Lip = np.linalg.eigvalsh(Cxx)[-1] + lam2; thr = lam1 / Lip
    tcol = types[None, :]
    A = np.zeros((N, N)); Z = A.copy(); tk = 1.0
    for _ in range(n_iter):
        V = Z - (Z @ Cxx - Cyx + lam2 * Z) / Lip
        V = np.sign(V) * np.maximum(np.abs(V) - thr, 0.0)
        V = tcol * np.maximum(tcol * V, 0.0)
        tnew = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        Z = V + ((tk - 1) / tnew) * (V - A); A, tk = V, tnew
    np.fill_diagonal(A, 0.0)
    return A


def dale_cleanup(A):
    N = A.shape[0]; B = A.copy(); t = strongest_entry_types(A)
    for j in range(N):
        col = B[:, j]; col[np.sign(col) != t[j]] = 0.0; B[:, j] = col
    np.fill_diagonal(B, 0.0)
    return B


def mixture_threshold(A):
    from sklearn.mixture import GaussianMixture
    N = A.shape[0]; off = ~np.eye(N, dtype=bool)
    x = A[off].reshape(-1, 1)
    gm = GaussianMixture(n_components=3, random_state=0, n_init=2).fit(x)
    unconn = int(np.argmin(np.abs(gm.means_.ravel())))
    lab = gm.predict(x); B = A.copy(); flat = B[off].copy()
    flat[lab == unconn] = 0.0; B[off] = flat; np.fill_diagonal(B, 0.0)
    return B


def balance_g(t, rates):
    E, I = t > 0, t < 0
    if E.sum() == 0 or I.sum() == 0 or rates[I].mean() == 0:
        return np.nan
    return (E.sum() * rates[E].mean()) / (I.sum() * rates[I].mean())


def rescale_balance_nz(A, rates):
    N = A.shape[0]; off = ~np.eye(N, dtype=bool); B = A.copy()
    t = strongest_entry_types(A); g = balance_g(t, rates)
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


def per_class_inhibitory(A, adj):
    """Inhibitory-class precision & recall (one-vs-rest) from the 3-class labels."""
    yt, yp = M._ternary_labels(A, adj)
    c = -1
    tp = int(((yp == c) & (yt == c)).sum())
    fp = int(((yp == c) & (yt != c)).sum())
    fn = int(((yp != c) & (yt == c)).sum())
    p = tp / (tp + fp) if (tp + fp) else float("nan")
    r = tp / (tp + fn) if (tp + fn) else float("nan")
    return p, r


def main():
    adj = np.load(Path(DS) / "adj_true.npy"); np.fill_diagonal(adj, 0.0)
    feed = np.asarray(zarr.open(FEED, "r")[:])
    meta = json.loads((Path(DS) / "metadata.json").read_text())["sim_params"]
    spk = np.asarray(zarr.open(str(Path(DS) / "spikes.zarr"), "r")[:])
    rates = spk.sum(1) / (meta["sim_time"] / 1000.0)
    N = adj.shape[0]; off = ~np.eye(N, dtype=bool); m = connectivity_metrics._metrics

    Xc = feed[:, :-LAG] - feed[:, :-LAG].mean(1, keepdims=True)
    Yc = feed[:, LAG:] - feed[:, LAG:].mean(1, keepdims=True)
    A_ols = (Yc @ Xc.T) @ np.linalg.inv((Xc @ Xc.T) + 1e-9 * np.eye(N)); np.fill_diagonal(A_ols, 0.0)
    A_en = fista_solve(FEED, lag=LAG, lam_l1=LAM1, lam_l2=LAM2, n_iter=500, chunk_size=10000)
    np.fill_diagonal(A_en, 0.0)
    A_dale = dale_fista(Xc, Yc, strongest_entry_types(A_en))
    A_full = rescale_balance_nz(A_dale, rates)
    A_mix = rescale_balance_nz(mixture_threshold(dale_cleanup(A_en)), rates)
    methods = [("OLS", A_ols), ("EN(3e-3)", A_en), ("EN+Dale", A_dale),
               ("EN+Dale+balance", A_full), ("EN+dale+mix+balance", A_mix)]

    def ei(A):
        g = adj.T[off]; a = np.abs(A[off]); ih, ex = a[g < 0], a[g > 0]
        return float(np.median(ih) / np.median(ex)) if len(ih) and len(ex) and np.median(ex) else np.nan

    cols = ["pears", "spear", "AUC", "F1bin", "mP", "mR", "mF1", "inhP", "inhR", "typeAcc", "dale", "E/I"]
    print("MAGNITUDE: pears/spear  |  DETECTION: AUC,F1bin (binary)  |  3-CLASS: mP/mR/mF1 (macro),")
    print("inhP/inhR (inhibitory one-vs-rest)  |  TYPE: typeAcc,dale  |  MAGNITUDE ratio: E/I (true=5)\n")
    print(f"{'method':>20} " + "".join(f"{c:>8}" for c in cols))
    print("-" * (21 + 8 * len(cols)))
    for name, A in methods:
        ip, ir = per_class_inhibitory(A, adj)
        row = dict(pears=m["pearson"](A, adj), spear=m["spearman"](A, adj), AUC=m["auc_roc"](A, adj),
                   F1bin=m["f1"](A, adj), mP=m["macro_precision"](A, adj), mR=m["macro_recall"](A, adj),
                   mF1=m["macro_f1"](A, adj), inhP=ip, inhR=ir,
                   typeAcc=m["dale_type_accuracy"](A, adj), dale=m["degree_of_daleianity"](A, adj), **{"E/I": ei(A)})
        print(f"{name:>20} " + "".join(f"{row[c]:>8.3f}" for c in cols))


if __name__ == "__main__":
    main()
