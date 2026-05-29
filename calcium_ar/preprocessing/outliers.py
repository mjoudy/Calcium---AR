"""
Robust linear regression methods for phase-space fitting (signal vs. derivative).

All public functions accept either:
  - A single 1-D pair (x, y)
  - A 2-D array or list of 1-D arrays (one dataset per neuron)

and return the fitted slope(s).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Internal dispatcher
# ---------------------------------------------------------------------------

def _is_multiset(x) -> bool:
    if isinstance(x, list):
        return True
    if isinstance(x, np.ndarray):
        return x.ndim == 2 or x.dtype == object
    return False


def _dispatch(signal, deriv, core_fn, **kwargs) -> float | np.ndarray:
    """Apply core_fn over each neuron dataset; handle 1-D vs multi-set."""
    multi = _is_multiset(signal)
    xs = signal if multi else [np.asarray(signal)]
    ys = deriv if multi else [np.asarray(deriv)]
    do_plot = kwargs.get("do_plot", False)
    results = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        x, y = np.asarray(x), np.asarray(y)
        kwargs["do_plot"] = do_plot and (i == 0)
        results.append(core_fn(x, y, **kwargs) if len(x) >= 2 else 0.0)
    out = np.array(results)
    return out[0] if not multi else out


# ---------------------------------------------------------------------------
# Plain least-squares fit
# ---------------------------------------------------------------------------

def fit_linear(x, y, do_plot=False, xlabel="Signal", ylabel="Derivative"):
    """Ordinary least-squares slope (y = slope·x + intercept)."""
    def _core(x, y, do_plot=False, **_):
        slope, intercept = np.polyfit(x, y, 1)
        if do_plot:
            fig, ax = plt.subplots()
            ax.scatter(x, y, s=5, marker=".")
            ax.plot(x, intercept + slope * x, color="k")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        return slope
    return _dispatch(x, y, _core, do_plot=do_plot)


# ---------------------------------------------------------------------------
# IQR
# ---------------------------------------------------------------------------

def remove_outliers_iqr(signal, deriv, threshold=1.5, p_lo=25, p_hi=75, do_plot=False):
    """Remove residual outliers by IQR fencing, then refit."""
    def _core(x, y, threshold=1.5, p_lo=25, p_hi=75, do_plot=False):
        s0, b0 = np.polyfit(x, y, 1)
        res = y - (s0 * x + b0)
        q1, q3 = np.percentile(res, [p_lo, p_hi])
        iqr = q3 - q1
        mask = (res >= q1 - threshold * iqr) & (res <= q3 + threshold * iqr)
        if mask.sum() < 2:
            return 0.0
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        if do_plot:
            fig, ax = plt.subplots()
            ax.scatter(x[mask], y[mask], s=5, marker=".")
            ax.plot(x[mask], intercept + slope * x[mask], color="k")
        return slope
    return _dispatch(signal, deriv, _core, threshold=threshold, p_lo=p_lo, p_hi=p_hi, do_plot=do_plot)


# ---------------------------------------------------------------------------
# RANSAC
# ---------------------------------------------------------------------------

def remove_outliers_ransac(signal, deriv, residual_threshold=0.05, do_plot=False):
    """RANSAC robust regression — ignores spike-trajectory outliers."""
    def _core(x, y, residual_threshold=0.05, do_plot=False):
        ransac = RANSACRegressor(residual_threshold=residual_threshold)
        try:
            ransac.fit(x.reshape(-1, 1), y)
        except Exception:
            return 0.0
        slope = ransac.estimator_.coef_[0]
        intercept = ransac.estimator_.intercept_
        if do_plot:
            inliers = ransac.inlier_mask_
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(x[~inliers], y[~inliers], s=2, c="lightgrey", label="Outliers")
            ax.scatter(x[inliers], y[inliers], s=2, c="seagreen", label="Inliers")
            xr = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
            ax.plot(xr, ransac.predict(xr), color="k", lw=2, label="RANSAC fit")
            ax.set_xlabel("Calcium signal")
            ax.set_ylabel("Derivative")
            ax.legend()
        return slope
    return _dispatch(signal, deriv, _core, residual_threshold=residual_threshold, do_plot=do_plot)


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------

def remove_outliers_dbscan(signal, deriv, eps=0.15, min_samples=30, do_plot=False):
    """Identify the largest density cluster (bulk decay) and fit a line to it."""
    def _core(x, y, eps=0.15, min_samples=30, do_plot=False):
        data = StandardScaler().fit_transform(np.column_stack((x, y)))
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit(data).labels_
        valid = labels[labels != -1]
        if len(valid) == 0:
            return 0.0
        bulk = np.bincount(valid).argmax()
        mask = labels == bulk
        model = LinearRegression().fit(x[mask].reshape(-1, 1), y[mask])
        slope = model.coef_[0]
        if do_plot:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(x[~mask], y[~mask], s=2, c="lightgrey", label="Non-bulk")
            ax.scatter(x[mask], y[mask], s=2, c="royalblue", label="Bulk")
            xr = np.array([x.min(), x.max()])
            ax.plot(xr, model.predict(xr.reshape(-1, 1)), color="k", lw=2)
            ax.set_xlabel("Calcium signal")
            ax.set_ylabel("Derivative")
            ax.legend()
        return slope
    return _dispatch(signal, deriv, _core, eps=eps, min_samples=min_samples, do_plot=do_plot)


# ---------------------------------------------------------------------------
# Z-score
# ---------------------------------------------------------------------------

def remove_outliers_zscore(signal, deriv, threshold=2.0, do_plot=False):
    """Remove points beyond `threshold` standard deviations, then refit."""
    def _core(x, y, threshold=2.0, do_plot=False):
        zx = (x - x.mean()) / (x.std() + 1e-9)
        zy = (y - y.mean()) / (y.std() + 1e-9)
        mask = (np.abs(zx) < threshold) & (np.abs(zy) < threshold)
        if mask.sum() < 2:
            return 0.0
        slope, _ = np.polyfit(x[mask], y[mask], 1)
        return slope
    return _dispatch(signal, deriv, _core, threshold=threshold, do_plot=do_plot)
