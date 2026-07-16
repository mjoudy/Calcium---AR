"""
Wrap-up figures. Reads the per-seed estimates written by scripts/wrapup_run.py
and produces the two presentable figures:

  Figure 1  method_panel.png       -- per method: 3x3 confusion (mean over seeds)
                                      at the 10% density threshold, ECDF of
                                      inferred weights by true class, and a
                                      combined bar chart of the reported metrics
                                      with seed error bars.
  Figure 2  decision_<method>.png  -- one method (default EN+Dale): the ROC and PR
                                      curves with three operating points marked
                                      (OP_DENS, default 3/6/9% density), and the
                                      3x3 confusion matrix produced at each point,
                                      so the curve <-> confusion link is explicit.

Shown metrics (only these, by request): correlation, ROC (+AUC), PR (+AP),
3x3 confusion, per-class precision/recall, macro precision/recall.

Usage:  python scripts/wrapup_figures.py --data results/wrapup_local
        python scripts/wrapup_figures.py --data $WS/results/wrapup_n1250 \
               --decision-method endale
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from calcium_ar.experiments import thresholding as T

DATA = ROOT / "results" / "wrapup_local"   # overridden by --data
DENSITY = 0.10
OP_DENS = [0.03, 0.06, 0.09]          # Figure 2 operating points (within EN+Dale support)
OP_LETTERS = ["A", "B", "C"]


def op_labels():
    return [f"{L}  ({d*100:.0f}%)" for L, d in zip(OP_LETTERS, OP_DENS)]

# --- method order + colours (validated categorical palette, fixed order) ----- #
METHODS = [("ols", "OLS", "#2a78d6"),
           ("en", "EN", "#1baf7a"),
           ("lasso", "Lasso", "#eda100"),
           ("lassodale", "Lasso+Dale", "#008300"),
           ("endale", "EN+Dale", "#4a3aa7")]

# class colours: E=blue, none=gray, I=red (diverging poles + neutral)
CLS_COLOR = {"E": "#2a78d6", "none": "#898781", "I": "#e34948"}

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
BLUE_SEQ = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fe", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.titlesize": 9.5, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

# metrics shown in the bar chart (key, label). Two families, split by a divider:
# threshold-FREE ranking + typing, then detection AT the 10% density threshold.
BAR_FREE = [("corr", "corr"), ("roc_auc", "ROC AUC"), ("pr_ap", "PR AP"),
            ("auc_E", "AUC E"), ("auc_I", "AUC I"), ("type_acc", "type acc")]
BAR_THR = [("macro_p", "macro P"), ("macro_r", "macro R"),
           ("E_p", "E prec"), ("E_r", "E rec"),
           ("I_p", "I prec"), ("I_r", "I rec")]
BAR_METRICS = BAR_FREE + BAR_THR


def seed_dirs():
    return sorted([p for p in DATA.glob("seed*") if (p / "adj_true.npy").exists()])


def load():
    """Return dict method -> list of (A, adj_true) over seeds."""
    seeds = seed_dirs()
    if not seeds:
        raise SystemExit(f"No seed data under {DATA}. Run scripts/wrapup_run.py first.")
    out = {m[0]: [] for m in METHODS}
    for sd in seeds:
        adj = np.load(sd / "adj_true.npy")
        for key, _, _ in METHODS:
            A = np.load(sd / f"A_{key}.npy")
            out[key].append((A, adj))
    return out, len(seeds)


# --------------------------------------------------------------------------- #
# Figure 1                                                                    #
# --------------------------------------------------------------------------- #

def confusion_mean(pairs, density):
    """Mean row-normalised 3x3 confusion over seeds (rows=true E/none/I)."""
    mats = []
    for A, adj in pairs:
        M = T.confusion3(A, adj, density=density).astype(float)
        M = M / np.clip(M.sum(1, keepdims=True), 1, None)
        mats.append(M)
    return np.mean(mats, axis=0)


def draw_confusion(ax, Mn, title):
    ax.imshow(Mn, cmap=BLUE_SEQ, vmin=0, vmax=1, aspect="equal")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{Mn[i, j]:.2f}", ha="center", va="center",
                    color="white" if Mn[i, j] > 0.55 else INK, fontsize=8)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(T.CLASS_NAMES); ax.set_yticklabels(T.CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)


def draw_ecdf(ax, pairs, density):
    """ECDF of signed inferred weights, one curve per true class, pooled seeds."""
    pooled = {"E": [], "none": [], "I": []}
    thr_vals = []
    for A, adj in pairs:
        g, a, a_abs = T._offdiag(A, adj)
        pooled["E"].append(a[g > 0]); pooled["I"].append(a[g < 0])
        pooled["none"].append(a[g == 0])
        m = T.connected_mask_at_density(a_abs, density)
        if m.any():                          # guard: a collapsed (all-zero) method
            thr_vals.append(a_abs[m].min())  # smallest |w| still called connected
    thr = float(np.mean(thr_vals)) if thr_vals else 0.0
    allw = np.concatenate([a for A, adj in pairs
                           for a in [T._offdiag(A, adj)[1]]])
    lo, hi = np.percentile(allw, [1, 99])
    for name in ("I", "none", "E"):
        v = np.sort(np.concatenate(pooled[name]))
        if v.size == 0:
            continue
        y = np.arange(1, v.size + 1) / v.size
        ax.plot(v, y, color=CLS_COLOR[name], lw=1.8, label=name)
    ax.axvline(thr, color=MUTED, ls="--", lw=0.9)
    ax.axvline(-thr, color=MUTED, ls="--", lw=0.9)
    ax.set_xlim(lo, hi); ax.set_ylim(0, 1)
    ax.set_xlabel("inferred weight"); ax.set_ylabel("CDF")
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def draw_bars(ax, scores, n_seeds):
    """Grouped bar chart: metric groups on x, one bar per method, seed std bars."""
    n_m = len(METHODS)
    gw = 0.8
    bw = gw / n_m
    x = np.arange(len(BAR_METRICS))
    for mi, (key, label, color) in enumerate(METHODS):
        means = [scores[key][k][0] for k, _ in BAR_METRICS]
        stds = [scores[key][k][1] for k, _ in BAR_METRICS]
        off = -gw / 2 + bw * (mi + 0.5)
        ax.bar(x + off, means, bw * 0.92, yerr=stds, color=color, label=label,
               error_kw=dict(lw=0.8, ecolor=MUTED, capsize=1.5))
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in BAR_METRICS])
    ax.set_ylim(0, 1.0); ax.set_ylabel("score")
    ax.grid(True, axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    # divider + family labels: threshold-free ranking/typing vs detection at 10%
    div = len(BAR_FREE) - 0.5
    ax.axvline(div, color=MUTED, lw=0.8, ls=":")
    ax.text((div - 0.5) / 2 / len(BAR_METRICS), 1.02, "threshold-free (ranking + typing)",
            transform=ax.transAxes, ha="center", va="bottom", color=MUTED, fontsize=8.5)
    ax.text((div + (len(BAR_METRICS) - div) / 2) / len(BAR_METRICS), 1.02,
            f"at {int(DENSITY*100)}% density (one operating point)",
            transform=ax.transAxes, ha="center", va="bottom", color=MUTED, fontsize=8.5)
    ax.legend(ncol=n_m, loc="upper center", bbox_to_anchor=(0.5, 1.20),
              frameon=False, fontsize=8.5)
    ax.set_title(f"reported metrics  (mean ± std over {n_seeds} seeds)",
                 color=INK, pad=32)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def figure1(data, n_seeds):
    # per-method, per-metric mean/std
    scores = {}
    for key, _, _ in METHODS:
        per_seed = [T.score_all(A, adj, density=DENSITY) for A, adj in data[key]]
        agg = {}
        for k, _ in BAR_METRICS:
            vals = np.array([ps[k] for ps in per_seed], dtype=float)
            agg[k] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
        scores[key] = agg

    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(3, len(METHODS), height_ratios=[1.05, 1.0, 1.25],
                          hspace=0.55, wspace=0.35,
                          left=0.06, right=0.98, top=0.9, bottom=0.08)
    for mi, (key, label, color) in enumerate(METHODS):
        ax_c = fig.add_subplot(gs[0, mi])
        draw_confusion(ax_c, confusion_mean(data[key], DENSITY), label)
        ax_e = fig.add_subplot(gs[1, mi])
        draw_ecdf(ax_e, data[key], DENSITY)
        if mi == 0:
            ax_e.legend(frameon=False, fontsize=8, loc="lower right",
                        title="true class")
    ax_b = fig.add_subplot(gs[2, :])
    draw_bars(ax_b, scores, n_seeds)

    N = data["ols"][0][0].shape[0]
    fig.suptitle(f"Figure 1 - Method comparison (N={N}, {n_seeds} seeds)",
                 fontsize=13, color=INK, x=0.06, ha="left", y=0.975)
    out = DATA / "method_panel.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Figure 2 - decision figure (EN+Dale)                                        #
# --------------------------------------------------------------------------- #

def mean_curve(pairs, kind):
    """Mean ROC or PR curve over seeds on a common grid."""
    grid = np.linspace(0, 1, 200)
    ys = []
    for A, adj in pairs:
        if kind == "roc":
            fpr, tpr, auc = T.roc_curve(A, adj)
            ys.append(np.interp(grid, fpr, tpr)); score = auc
        else:
            prec, rec, ap = T.pr_curve(A, adj)
            order = np.argsort(rec)
            ys.append(np.interp(grid, rec[order], prec[order])); score = ap
    return grid, np.mean(ys, axis=0), score


def op_points(pairs, density, kind):
    """Mean operating point (x, y) at a density: PR->(recall,prec), ROC->(fpr,tpr)."""
    xs, yss = [], []
    for A, adj in pairs:
        g, a, a_abs = T._offdiag(A, adj)
        y_true = (g != 0).astype(int)
        pred = T.connected_mask_at_density(a_abs, density).astype(int)
        tp = int((pred & y_true).sum()); fp = int((pred & (1 - y_true)).sum())
        fn = int(((1 - pred) & y_true).sum()); tn = int(((1 - pred) & (1 - y_true)).sum())
        if kind == "pr":
            xs.append(tp / (tp + fn) if tp + fn else 0)      # recall
            yss.append(tp / (tp + fp) if tp + fp else 0)     # precision
        else:
            xs.append(fp / (fp + tn) if fp + tn else 0)      # fpr
            yss.append(tp / (tp + fn) if tp + fn else 0)     # tpr
    return float(np.mean(xs)), float(np.mean(yss))


def figure2(data, method_key="endale"):
    label = dict(METHODS and [(m[0], m[1]) for m in METHODS])[method_key]
    pairs = data[method_key]
    dot_c = ["#4a3aa7", "#e34948", "#eb6834"]

    fig = plt.figure(figsize=(13.5, 6.2))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.25, 1.25, 1.1],
                          hspace=0.55, wspace=0.4,
                          left=0.06, right=0.985, top=0.86, bottom=0.11)
    ax_pr = fig.add_subplot(gs[:, 0])
    ax_roc = fig.add_subplot(gs[:, 1])

    # PR curve
    gr, pr_m, ap = mean_curve(pairs, "pr")
    ax_pr.plot(gr, pr_m, color="#2a78d6", lw=2)
    ax_pr.fill_between(gr, 0, pr_m, color="#2a78d6", alpha=0.10)
    for d, c, lab in zip(OP_DENS, dot_c, op_labels()):
        x, y = op_points(pairs, d, "pr")
        ax_pr.plot(x, y, "o", ms=9, color=c, mec="white", mew=1.2, zorder=5)
        ax_pr.annotate(lab.split()[0], (x, y), textcoords="offset points",
                       xytext=(7, 6), color=c, fontweight="bold", fontsize=9)
    ax_pr.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="recall", ylabel="precision",
              title=f"PR curve  (AP = {ap:.2f})")

    # ROC curve
    gr, roc_m, auc = mean_curve(pairs, "roc")
    ax_roc.plot(gr, roc_m, color="#2a78d6", lw=2)
    ax_roc.fill_between(gr, 0, roc_m, color="#2a78d6", alpha=0.10)
    ax_roc.plot([0, 1], [0, 1], ls=":", color=MUTED, lw=1)
    for d, c, lab in zip(OP_DENS, dot_c, op_labels()):
        x, y = op_points(pairs, d, "roc")
        ax_roc.plot(x, y, "o", ms=9, color=c, mec="white", mew=1.2, zorder=5)
        ax_roc.annotate(lab.split()[0], (x, y), textcoords="offset points",
                        xytext=(7, -12), color=c, fontweight="bold", fontsize=9)
    ax_roc.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="false positive rate",
               ylabel="true positive rate", title=f"ROC curve  (AUC = {auc:.2f})")

    for ax in (ax_pr, ax_roc):
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # three confusion matrices, one per operating point
    for i, (d, c, lab) in enumerate(zip(OP_DENS, dot_c, op_labels())):
        ax = fig.add_subplot(gs[i, 2])
        draw_confusion(ax, confusion_mean(pairs, d), "")
        ax.set_title(lab, color=c, fontweight="bold", loc="left", fontsize=9)
        if i < 2:
            ax.set_xlabel("")

    N = pairs[0][0].shape[0]
    fig.suptitle("Figure 2 - Decision making: the threshold walks the curve "
                 f"({label}, N={N})", fontsize=13, color=INK, x=0.06, ha="left")
    out = DATA / f"decision_{method_key}.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Figure 3 - excitatory threshold trade-off                                   #
# --------------------------------------------------------------------------- #
# Inhibitory is already saturated (recall ~1.0 at any threshold), so the only
# class the threshold actually moves is excitatory. This figure isolates it:
# as you lower the excitatory (positive-side) threshold you recover more true E
# (missed E / false negatives fall) but wrongly promote more none-edges to E
# (false positives rise). Left panel shows that count trade-off directly; right
# panel is the same thing as the excitatory precision-recall curve.

def _exc_curves(pairs, grid_dens):
    """Mean over seeds of TP/FP/FN counts and precision/recall for E-vs-rest,
    as a function of predicted-excitatory density (positive-side threshold)."""
    TP, FP, FN, PR, RC = [], [], [], [], []
    for A, adj in pairs:
        g, a, _ = _off(A, adj)
        nE = int((g > 0).sum())
        ntot = len(g)
        order = np.argsort(a)[::-1]            # most positive first
        isE = (g[order] > 0)
        cumE = np.cumsum(isE)                  # true E among the top-k positives
        ks = np.clip((grid_dens * ntot).astype(int), 1, ntot)
        tp = cumE[ks - 1].astype(float)
        fp = ks.astype(float) - tp
        fn = nE - tp
        TP.append(tp); FP.append(fp); FN.append(fn)
        PR.append(tp / np.maximum(tp + fp, 1)); RC.append(tp / max(nE, 1))
    return (np.mean(TP, 0), np.mean(FP, 0), np.mean(FN, 0),
            np.mean(PR, 0), np.mean(RC, 0))


def _exc_op_at_total(pairs, density):
    """Excitatory operating point produced by the |w| density threshold used in
    scoring: returns mean (predicted-E density, recall, precision)."""
    pe, rc, pr = [], [], []
    for A, adj in pairs:
        g, a, a_abs = _off(A, adj)
        conn = T.connected_mask_at_density(a_abs, density)
        predE = conn & (a > 0)
        nP = int(predE.sum()); tp = int((predE & (g > 0)).sum())
        pe.append(nP / len(g))
        rc.append(tp / max(int((g > 0).sum()), 1))
        pr.append(tp / max(nP, 1))
    return float(np.mean(pe)), float(np.mean(rc)), float(np.mean(pr))


def _off(A, adj):
    return T._offdiag(A, adj)


def figure_excitatory(data, method_key="endale"):
    label = dict((m[0], m[1]) for m in METHODS)[method_key]
    pairs = data[method_key]
    grid = np.linspace(0.003, 0.20, 120)
    TP, FP, FN, PRc, RCc = _exc_curves(pairs, grid)
    op_d, op_r, op_p = _exc_op_at_total(pairs, DENSITY)

    C_FN, C_FP = "#2a78d6", "#eb6834"          # missed-E (blue), false-E (orange)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.13, wspace=0.28)

    # Left: count trade-off vs predicted-E density
    axL.plot(grid * 100, FN, color=C_FN, lw=2, label="missed true E  (false neg)")
    axL.plot(grid * 100, FP, color=C_FP, lw=2, label="none called E  (false pos)")
    axL.axvline(op_d * 100, color=MUTED, ls="--", lw=1)
    axL.annotate(f"10% density\noperating point\n(E recall {op_r:.2f})",
                 (op_d * 100, axL.get_ylim()[1]), textcoords="offset points",
                 xytext=(6, -4), va="top", color=MUTED, fontsize=8.5)
    axL.set(xlabel="predicted excitatory density (%)   →  looser threshold",
            ylabel="edge count", title="excitatory error trade-off")
    axL.legend(frameon=False, fontsize=9, loc="upper right")
    axL.grid(True, color=GRID, lw=0.6); axL.set_axisbelow(True)

    # Right: excitatory precision-recall curve
    axR.plot(RCc, PRc, color="#2a78d6", lw=2)
    axR.plot(op_r, op_p, "o", ms=10, color="#4a3aa7", mec="white", mew=1.2, zorder=5)
    axR.annotate("10% density", (op_r, op_p), textcoords="offset points",
                 xytext=(8, 6), color="#4a3aa7", fontweight="bold", fontsize=9)
    axR.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="excitatory recall",
            ylabel="excitatory precision", title="excitatory precision-recall")
    axR.grid(True, color=GRID, lw=0.6); axR.set_axisbelow(True)

    for ax in (axL, axR):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    N = pairs[0][0].shape[0]
    fig.suptitle("Figure 3 - Excitatory threshold trade-off  "
                 f"({label}, N={N}; inhibitory is already ~perfect)",
                 fontsize=13, color=INK, x=0.07, ha="left")
    out = DATA / f"excitatory_{method_key}.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


def main():
    import argparse
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="dir with seed*/ estimates (default: results/wrapup_local)")
    ap.add_argument("--decision-method", default="endale",
                    choices=[m[0] for m in METHODS])
    ap.add_argument("--decision-compare", action="store_true",
                    help="also render the OLS decision figure for comparison")
    args = ap.parse_args()
    if args.data:
        DATA = Path(args.data)

    data, n_seeds = load()
    f1 = figure1(data, n_seeds)
    f2 = figure2(data, args.decision_method)
    f3 = figure_excitatory(data, args.decision_method)
    print(f"wrote {f1}")
    print(f"wrote {f2}")
    print(f"wrote {f3}")
    if args.decision_compare and args.decision_method != "ols":
        print(f"wrote {figure2(data, 'ols')}")


if __name__ == "__main__":
    main()
