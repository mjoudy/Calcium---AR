"""
Density-based thresholding and the reported metric set for the wrap-up.

This module is *additive* — it does not replace anything in `metrics.py`. It
implements the specific scoring rule chosen for the wrap-up figures:

    Before every threshold-dependent measurement, pick the threshold that makes
    the predicted connection density equal a target (default 10%, which equals
    the true Brunel density epsilon=0.1). Concretely: rank the off-diagonal
    edges by |inferred weight| and keep the top `density` fraction as
    "connected". The sign of the kept weight assigns the 3-class label
    (+1 excitatory / -1 inhibitory / 0 unconnected).

Two metric families:

  threshold-free  — correlation (Pearson), ROC (+AUC), PR (+AP). Computed on the
                    continuous weights; NO thresholding (that is the whole point).
  threshold-dependent — 3x3 confusion, per-class precision/recall, macro
                    precision/recall. Computed at the density threshold.

Convention (identical to metrics.py):
    adj_inferred[i, j]  <->  adj_true.T[i, j]
    i.e. inferred weight j->i is compared against true weight j->i (source->target).
    All functions below compare against adj_true.T and exclude the diagonal.

Sparse methods (pure Lasso, Lasso+Dale) also expose a "native" scoring where the
solver's own exact zeros define the unconnected class — so a method that picks
fewer edges than the target density is judged on what it actually claimed.
"""

from __future__ import annotations

import numpy as np

# Class order used everywhere for confusion rows/cols and per-class dicts.
CLASSES = (1, 0, -1)            # excitatory, unconnected, inhibitory
CLASS_NAMES = ("E", "none", "I")


# --------------------------------------------------------------------------- #
# Off-diagonal extraction (source->target convention)                         #
# --------------------------------------------------------------------------- #

def _offdiag(adj_inferred: np.ndarray, adj_true: np.ndarray):
    """Return flattened off-diagonal (true_signed, inferred_signed, |inferred|)."""
    N = adj_true.shape[0]
    mask = ~np.eye(N, dtype=bool)
    g = adj_true.T[mask].ravel()          # transpose = source->target convention
    a = adj_inferred[mask].ravel()
    return g, a, np.abs(a)


# --------------------------------------------------------------------------- #
# Thresholding                                                                #
# --------------------------------------------------------------------------- #

def connected_mask_at_density(a_abs: np.ndarray, density: float) -> np.ndarray:
    """Boolean mask over edges: keep those with |weight| above the score
    threshold that yields the target predicted `density`.

    The threshold is the (1 - density) quantile of |weight|. Using a *score*
    threshold (not a top-K rank) keeps the binary "connected" decision and the
    3-class sign decision consistent: a selected edge always has a non-zero
    weight, so its sign (E/I) is well defined and it lies on the ROC/PR curve.

    For a sparse method whose non-zero support is already below the target
    density, the quantile is zero; we then fall back to the native support
    (|weight| > 0). Such a method simply cannot be pushed to the target density
    -- there is no ranking information among its zeros -- and this exposes that
    honestly instead of padding with arbitrary zero-weight edges.
    """
    tau = float(np.quantile(a_abs, 1.0 - density))
    if tau <= 0.0:
        return a_abs > 0.0
    return a_abs > tau


def _predict_ternary(a_signed, a_abs, density):
    """Predicted 3-class labels at the density threshold (+1/0/-1)."""
    conn = connected_mask_at_density(a_abs, density)
    y_pred = np.where(conn, np.sign(a_signed), 0).astype(int)
    return y_pred


def _predict_ternary_native(a_signed, a_abs, eps=1e-12):
    """Predicted 3-class labels from the solver's own exact zeros."""
    return np.where(a_abs > eps, np.sign(a_signed), 0).astype(int)


# --------------------------------------------------------------------------- #
# Threshold-free metrics                                                      #
# --------------------------------------------------------------------------- #

def correlation(adj_inferred, adj_true) -> float:
    """Pearson correlation of signed off-diagonal weights (true vs inferred)."""
    g, a, _ = _offdiag(adj_inferred, adj_true)
    if a.std() == 0 or g.std() == 0:
        return 0.0
    return float(np.corrcoef(a, g)[0, 1])


def roc_curve(adj_inferred, adj_true):
    """(fpr, tpr, auc) for detecting existence of a connection (score=|weight|)."""
    from sklearn.metrics import roc_curve as _roc, roc_auc_score
    g, _, s = _offdiag(adj_inferred, adj_true)
    y = (g != 0).astype(int)
    if y.sum() in (0, len(y)):
        return np.array([0, 1]), np.array([0, 1]), float("nan")
    fpr, tpr, _ = _roc(y, s)
    return fpr, tpr, float(roc_auc_score(y, s))


def pr_curve(adj_inferred, adj_true):
    """(precision, recall, average_precision) for connection detection."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    g, _, s = _offdiag(adj_inferred, adj_true)
    y = (g != 0).astype(int)
    if y.sum() == 0:
        return np.array([1, 1]), np.array([0, 1]), float("nan")
    prec, rec, _ = precision_recall_curve(y, s)
    return prec, rec, float(average_precision_score(y, s))


def auc_per_class(adj_inferred, adj_true):
    """Per-class one-vs-rest detection AUC (threshold-free), (auc_E, auc_I).

    Excitatory-vs-rest scores with the SIGNED weight (positive favours E);
    inhibitory-vs-rest scores with the negated weight (negative favours I).
    This is how the inhibitory class shows its true separability instead of
    being averaged with the harder excitatory class under one |w| ranking.
    """
    from sklearn.metrics import roc_auc_score
    g, a, _ = _offdiag(adj_inferred, adj_true)
    yE, yI = (g > 0).astype(int), (g < 0).astype(int)
    aucE = float(roc_auc_score(yE, a)) if 0 < yE.sum() < len(yE) else float("nan")
    aucI = float(roc_auc_score(yI, -a)) if 0 < yI.sum() < len(yI) else float("nan")
    return aucE, aucI


# --------------------------------------------------------------------------- #
# Threshold-dependent metrics                                                 #
# --------------------------------------------------------------------------- #

def confusion3(adj_inferred, adj_true, density=0.1, native=False):
    """3x3 confusion counts, rows = TRUE, cols = PRED, order (E, none, I).

    native=True scores against the solver's own zeros instead of the density
    threshold (only meaningful for sparse methods).
    """
    g, a, a_abs = _offdiag(adj_inferred, adj_true)
    y_true = np.sign(g).astype(int)
    y_pred = (_predict_ternary_native(a, a_abs) if native
              else _predict_ternary(a, a_abs, density))
    M = np.zeros((3, 3), dtype=int)
    for r, ct in enumerate(CLASSES):
        for c, cp in enumerate(CLASSES):
            M[r, c] = int(((y_true == ct) & (y_pred == cp)).sum())
    return M


def per_class_pr(adj_inferred, adj_true, density=0.1, native=False):
    """Per-class precision & recall (one-vs-rest) for E, none, I.

    Returns {'E': (p, r), 'none': (p, r), 'I': (p, r)}.
    """
    g, a, a_abs = _offdiag(adj_inferred, adj_true)
    y_true = np.sign(g).astype(int)
    y_pred = (_predict_ternary_native(a, a_abs) if native
              else _predict_ternary(a, a_abs, density))
    out = {}
    for cls, name in zip(CLASSES, CLASS_NAMES):
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        p = tp / (tp + fp) if (tp + fp) else float("nan")
        r = tp / (tp + fn) if (tp + fn) else float("nan")
        out[name] = (p, r)
    return out


def macro_pr(adj_inferred, adj_true, density=0.1, native=False):
    """Macro-averaged precision & recall over the 3 classes (each weighted equally)."""
    pc = per_class_pr(adj_inferred, adj_true, density=density, native=native)
    ps = [v[0] for v in pc.values() if not np.isnan(v[0])]
    rs = [v[1] for v in pc.values() if not np.isnan(v[1])]
    mp = float(np.mean(ps)) if ps else float("nan")
    mr = float(np.mean(rs)) if rs else float("nan")
    return mp, mr


# --------------------------------------------------------------------------- #
# One-call summary for a single (method, seed)                                #
# --------------------------------------------------------------------------- #

def score_all(adj_inferred, adj_true, density=0.1, native=False):
    """Flat dict of the reported scalar metrics for one estimate.

    Keys: corr, roc_auc, pr_ap, auc_E, auc_I (threshold-free ranking);
    type_acc (per-neuron E/I identification); macro_p, macro_r and per-class
    E_p/E_r/none_p/none_r/I_p/I_r (at the density threshold).  Curves are fetched
    separately via roc_curve()/pr_curve().
    """
    from .metrics import dale_type_accuracy
    corr = correlation(adj_inferred, adj_true)
    _, _, auc = roc_curve(adj_inferred, adj_true)
    _, _, ap = pr_curve(adj_inferred, adj_true)
    aucE, aucI = auc_per_class(adj_inferred, adj_true)
    type_acc = dale_type_accuracy(adj_inferred, adj_true)
    mp, mr = macro_pr(adj_inferred, adj_true, density=density, native=native)
    pc = per_class_pr(adj_inferred, adj_true, density=density, native=native)
    out = dict(corr=corr, roc_auc=auc, pr_ap=ap, auc_E=aucE, auc_I=aucI,
               type_acc=type_acc, macro_p=mp, macro_r=mr)
    for name in CLASS_NAMES:
        out[f"{name}_p"], out[f"{name}_r"] = pc[name]
    return out
