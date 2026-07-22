"""
R.1 (method-demonstration panel) — COMPUTE stage, runs ON THE CLUSTER.

Loads one estimate + ground truth, does all the heavy work on a large random
edge sample, and writes a tiny r1_data.npz (a few KB) holding only plot-ready
numbers: the 3x3 confusion, the ROC and PR curves (downsampled to a grid), the
per-class weight histograms and CDFs, and the scalar metrics. The local
fig_r1_plot.py renders it; nothing big needs to travel home.

Usage (on the cluster):
  python scripts/fig_r1_compute.py \
      --data $(ws_find calcium_ar)/results/wrapup_n12500_T5000k --method ols \
      --out  $(ws_find calcium_ar)/results/fig_data/r1_data.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

CLASSES = (1, 0, -1)           # E, none, I  (rows/cols of the confusion)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="run dir with A_<method>.npy + adj_true.npy")
    ap.add_argument("--seed", default="seed1")
    ap.add_argument("--method", default="ols")
    ap.add_argument("--density", type=float, default=0.10, help="operating point")
    ap.add_argument("--sample", type=int, default=20_000_000)
    ap.add_argument("--grid", type=int, default=256, help="points on the ROC/PR curves")
    ap.add_argument("--bins", type=int, default=80, help="histogram bins")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = Path(args.data) / args.seed
    adj = np.load(d / "adj_true.npy")               # (N,N) float32, diag already 0
    A = np.load(d / f"A_{args.method}.npy")
    N = adj.shape[0]

    # --- sample off-diagonal edges: position (i,j) -> true adj[j,i], est A[i,j] ---
    rng = np.random.default_rng(0)
    ii = rng.integers(0, N, args.sample); jj = rng.integers(0, N, args.sample)
    keep = ii != jj; ii, jj = ii[keep], jj[keep]
    g = np.asarray(adj[jj, ii], dtype=np.float64)
    a = np.asarray(A[ii, jj], dtype=np.float64)
    aa = np.abs(a)
    yc = (g != 0).astype(np.int8)
    yE, yI = g > 0, g < 0
    true_density = float(yc.mean())

    # --- operating point: |w| threshold at the target predicted density --------- #
    tau = float(np.quantile(aa, 1.0 - args.density))
    conn = aa > tau if tau > 0 else aa > 0
    y_true = np.sign(g).astype(int)
    y_pred = np.where(conn, np.sign(a), 0).astype(int)

    # --- 3x3 confusion (rows=true, cols=pred), row-normalised ------------------- #
    cm = np.zeros((3, 3))
    for r, ct in enumerate(CLASSES):
        for c, cp in enumerate(CLASSES):
            cm[r, c] = np.sum((y_true == ct) & (y_pred == cp))
    cm_norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)

    # --- per-class precision/recall + macro from the confusion counts ----------- #
    def prf(cls_row):
        tp = cm[cls_row, cls_row]
        fp = cm[:, cls_row].sum() - tp
        fn = cm[cls_row, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else np.nan
        r = tp / (tp + fn) if tp + fn else np.nan
        return p, r
    (Ep, Er), (Np, Nr), (Ip, Ir) = prf(0), prf(1), prf(2)
    macro_p = np.nanmean([Ep, Np, Ip]); macro_r = np.nanmean([Er, Nr, Ir])

    # --- ROC / PR curves, downsampled onto a fixed grid ------------------------- #
    fpr, tpr, _ = roc_curve(yc, aa)
    roc_auc = float(roc_auc_score(yc, aa))
    grid = np.linspace(0, 1, args.grid)
    tpr_g = np.interp(grid, fpr, tpr)
    prec, rec, _ = precision_recall_curve(yc, aa)
    order = np.argsort(rec)
    prec_g = np.interp(grid, rec[order], prec[order])
    from sklearn.metrics import average_precision_score
    pr_ap = float(average_precision_score(yc, aa))

    # operating point on both curves (from the confusion counts)
    TP = cm[0, 0] + cm[2, 2]                       # correct-sign connected
    P = cm[[0, 2], :].sum()                        # all true connected
    predP = (y_pred != 0).sum()
    Ntrue = (y_true == 0).sum()
    FP = predP - TP
    op_recall = TP / max(P, 1); op_prec = TP / max(predP, 1)
    op_fpr = FP / max(Ntrue, 1)

    # --- per-class detection AUC (E-vs-rest, I-vs-rest) ------------------------- #
    auc_E = float(roc_auc_score(yE.astype(int), a)) if yE.any() else np.nan
    auc_I = float(roc_auc_score(yI.astype(int), -a)) if yI.any() else np.nan
    corr = float(np.corrcoef(a, g)[0, 1])

    # --- neuron-type accuracy from the FULL columns (needs whole matrix) -------- #
    col_sum = np.asarray(A.sum(0), dtype=np.float64)   # inferred outgoing per neuron
    row_sum = np.asarray(adj.sum(1), dtype=np.float64) # true outgoing per neuron
    has_type = row_sum != 0
    type_acc = float(np.mean(np.sign(col_sum[has_type]) == np.sign(row_sum[has_type])))

    # --- weight distributions by true class (pdf + cdf) ------------------------- #
    lo, hi = np.percentile(a, [1, 99])
    edges = np.linspace(lo, hi, args.bins + 1)
    def hist(mask):
        return np.histogram(a[mask], bins=edges, density=True)[0]
    pdf_E, pdf_none, pdf_I = hist(yE), hist(g == 0), hist(yI)
    cy = np.linspace(0, 1, 400)
    def cdf_x(mask):
        v = a[mask]
        return np.quantile(v, cy) if v.size else np.full_like(cy, np.nan)
    cdf_E, cdf_none, cdf_I = cdf_x(yE), cdf_x(g == 0), cdf_x(yI)

    # --- label for titles ------------------------------------------------------- #
    label = Path(args.data).name        # e.g. wrapup_n12500_T5000k

    np.savez(
        args.out,
        # meta
        N=N, true_density=true_density, density=args.density, method=args.method,
        label=label, tau=tau, n_sampled=len(g),
        # confusion
        cm=cm, cm_norm=cm_norm,
        # curves
        grid=grid, tpr=tpr_g, prec=prec_g, roc_auc=roc_auc, pr_ap=pr_ap,
        op_fpr=op_fpr, op_recall=op_recall, op_prec=op_prec,
        # distributions
        edges=edges, pdf_E=pdf_E, pdf_none=pdf_none, pdf_I=pdf_I,
        cdf_y=cy, cdf_E=cdf_E, cdf_none=cdf_none, cdf_I=cdf_I,
        # scalar metrics
        corr=corr, auc_E=auc_E, auc_I=auc_I, type_acc=type_acc,
        macro_p=macro_p, macro_r=macro_r,
        E_p=Ep, E_r=Er, I_p=Ip, I_r=Ir, none_r=Nr,
    )
    print(f"wrote {args.out}  (N={N}, AUC={roc_auc:.3f}, E_rec={Er:.3f}, "
          f"I_rec={Ir:.3f}, type_acc={type_acc:.3f})")


if __name__ == "__main__":
    main()
