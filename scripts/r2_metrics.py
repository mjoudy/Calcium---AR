"""
R.2 connectivity metric set — shared by fig_r2_compute.py (score at sweep time)
and fig_r2_rescore.py (re-derive from a frozen OLS estimate, no re-sweep).

Thin wrapper over calcium_ar.experiments.thresholding so R.2 uses the SAME
source->target convention and the SAME density-quantile operating point as the
wrap-up figures (top `density` fraction of |A| off-diagonal entries counts as a
predicted edge; density=0.10 ~ true Brunel epsilon). That module imports only
numpy at load time, so the rescore path stays cheap.

recall_exc / recall_inh split the true edges by whether their SOURCE neuron
(column of A / row of adj) is excitatory (index < n_exc) or inhibitory —
recall_exc is the R.4 excitatory-recall gap number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calcium_ar.experiments.thresholding import _offdiag, connected_mask_at_density

# Order matters: this is the on-disk column order in r2_data.npz (<kind>_<name>).
METRICS = ("auc", "corr", "precision", "recall", "f1", "recall_exc", "recall_inh")


def metrics_from_A(A: np.ndarray, adj: np.ndarray, n_exc: int,
                   density: float = 0.10) -> dict[str, float]:
    """Score one OLS estimate A (N x N) against ground truth adj (N x N).

    auc, corr            threshold-free (unchanged from the original score()).
    precision/recall/f1  at the density operating point (connected_mask_at_density).
    recall_exc/_inh      recall over the true edges whose source neuron is
                         excitatory / inhibitory.
    """
    g, a, a_abs = _offdiag(A, adj)               # source->target, diagonal dropped
    gt = g != 0

    N = A.shape[0]
    src = np.broadcast_to(np.arange(N), (N, N))[~np.eye(N, dtype=bool)].ravel()
    exc_src = src < n_exc

    auc = float(roc_auc_score(gt.astype(int), a_abs))
    corr = float(np.corrcoef(a, g)[0, 1])

    pred = connected_mask_at_density(a_abs, density)
    tp = float((pred & gt).sum())
    fp = float((pred & ~gt).sum())
    fn = float((~pred & gt).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    def _recall(sub):
        gsub = gt & sub
        denom = float(gsub.sum())
        return float((pred & gsub).sum()) / denom if denom else float("nan")

    return dict(auc=auc, corr=corr, precision=precision, recall=recall, f1=f1,
                recall_exc=_recall(exc_src), recall_inh=_recall(~exc_src))
