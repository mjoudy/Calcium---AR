
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def fit_linear(x, y, do_plot=False, xlabel='X', ylabel='Y'):
    """
    Fit a linear model (y = ax + b).
    Unified logic: Treats input as list of datasets.
    """
    # Detect if multiple datasets (list or 2D) or single (1D)
    x = np.array(x, dtype=object) # Object allows ragged arrays if list passed
    y = np.array(y, dtype=object)
    
    # Heuristic:
    is_multiset = False
    if isinstance(x, list) or (isinstance(x, np.ndarray) and x.ndim == 1 and x.dtype == object):
        is_multiset = True
    elif isinstance(x, np.ndarray) and x.ndim == 2:
        is_multiset = True
        
    if not is_multiset:
        datasets_x = [x]
        datasets_y = [y]
    else:
        datasets_x = x
        datasets_y = y
        
    slopes = []
    
    for i in range(len(datasets_x)):
        # Handle each dataset
        cx = np.array(datasets_x[i])
        cy = np.array(datasets_y[i])
        
        if len(cx) < 2:
            slopes.append(0.0)
            continue
            
        s, intercept = np.polyfit(cx, cy, deg=1)
        slopes.append(s)
        
        if do_plot and i == 0:
            fig, ax = plt.subplots()
            ax.scatter(cx, cy, marker='.', s=5)
            ax.plot(cx, intercept + s * cx, color='k')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

    if not is_multiset:
        return slopes[0]
    return np.array(slopes)


def _remove_outliers_unified(signal, deriv, method_func, **kwargs):
    """
    Internal driver to handle 1D/2D/Ragged inputs uniformly.
    """
    # Determine input type (Single vs Multi)
    is_multiset = False
    if isinstance(signal, list):
         is_multiset = True
    elif isinstance(signal, np.ndarray):
         if signal.ndim == 2:
             is_multiset = True
         elif signal.dtype == object: # Ragged array from numpy
             is_multiset = True
             
    if not is_multiset:
        iter_sig = [np.array(signal)]
        iter_der = [np.array(deriv)]
    else:
        iter_sig = signal
        iter_der = deriv
        
    results = []
    do_plot = kwargs.get('do_plot', False)
    
    for i in range(len(iter_sig)):
        # Disable plot for subsequent items if batch
        kwargs['do_plot'] = do_plot and (i == 0)
        
        s = np.array(iter_sig[i])
        d = np.array(iter_der[i])
        
        if len(s) < 2:
            results.append(0.0)
        else:
            res = method_func(s, d, **kwargs)
            results.append(res)
            
    if not is_multiset:
        return results[0]
    return np.array(results)


def _iqr_core(x, y, threshold=1.5, percentile_start=25, percentile_end=75, do_plot=False):
    slope_init, intercept_init = np.polyfit(x, y, 1)
    residuals = y - (slope_init * x + intercept_init)
    
    quartile_1, quartile_3 = np.percentile(residuals, [percentile_start, percentile_end])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (threshold * iqr)
    upper_bound = quartile_3 + (threshold * iqr)
    
    mask = (residuals >= lower_bound) & (residuals <= upper_bound)
    inlier_x = x[mask]
    inlier_y = y[mask]
    
    if len(inlier_x) < 2: return 0.0
    
    slope, intercept = np.polyfit(inlier_x, inlier_y, deg=1)
    
    if do_plot:
        fig, ax = plt.subplots()
        ax.scatter(inlier_x, inlier_y, marker='.', s=5)
        ax.plot(inlier_x, intercept + slope * inlier_x, color='k')
    
    return slope

# Wrappers
def remove_outliers_iqr(signal, deriv, threshold=1.5, percentile_start=25, percentile_end=75, do_plot=False):
    return _remove_outliers_unified(signal, deriv, _iqr_core, threshold=threshold, percentile_start=percentile_start, percentile_end=percentile_end, do_plot=do_plot)

def remove_outliers_ransac(signal, deriv, do_plot=False):
    def _ransac_core(x, y, do_plot=False):
        # User specified residual_threshold=0.05
        ransac = RANSACRegressor(residual_threshold=0.05)
        x_reshaped = x.reshape(-1, 1)
        try:
            ransac.fit(x_reshaped, y)
            slope = ransac.estimator_.coef_[0]
            intercept = ransac.estimator_.intercept_
            
            if do_plot:
                inlier_mask = ransac.inlier_mask_
                outlier_mask = np.logical_not(inlier_mask)
                line_x = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
                line_y = ransac.predict(line_x)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(x[outlier_mask], y[outlier_mask], s=2, c='lightgrey', label='Non-Bulk (Outliers)')
                ax.scatter(x[inlier_mask], y[inlier_mask], s=2, c='seagreen', label='Bulk (Inliers)')
                ax.plot(line_x, line_y, color='black', linewidth=2, label='RANSAC Fit')
                ax.set_title("Method 3: RANSAC Robust Regression")
                ax.set_xlabel("Calcium Signal")
                ax.set_ylabel("Derivative")
                ax.legend()
                plt.show()

            return slope
        except:
            return 0.0
    return _remove_outliers_unified(signal, deriv, _ransac_core, do_plot=do_plot)


def remove_outliers_dbscan(signal, deriv, do_plot=False):
    def _dbscan_core(x, y, do_plot=False):
        try:
            # 1. Scale data
            data_matrix = np.column_stack((x, y))
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_matrix)
            
            # 2. DBSCAN
            # eps=0.15, min_samples=30 as requested
            db = DBSCAN(eps=0.15, min_samples=30).fit(data_scaled)
            labels = db.labels_
            
            # Identify largest non-noise cluster
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            if len(unique_labels) == 0:
                # Fallback if no clusters found
                return 0.0
                
            bulk_label = unique_labels[np.argmax(counts)]
            is_bulk = (labels == bulk_label)
            
            # 3. Fit Line
            model = LinearRegression().fit(x[is_bulk].reshape(-1, 1), y[is_bulk])
            slope = model.coef_[0]
            
            if do_plot:
                tau = -1/slope if slope != 0 else 0
                
                fig, ax = plt.subplots(figsize=(10, 6))
                # Plot non-bulk
                ax.scatter(x[~is_bulk], y[~is_bulk], s=2, c='lightgrey', label='Non-Bulk')
                # Plot bulk
                ax.scatter(x[is_bulk], y[is_bulk], s=2, c='royalblue', label='Bulk (DBSCAN)')
                
                x_range = np.array([x.min(), x.max()])
                y_pred = model.predict(x_range.reshape(-1, 1))
                ax.plot(x_range, y_pred, color='black', lw=2, label='Linear Fit')
                
                ax.set_title(f"Method 2: DBSCAN Results (Slope={slope:.6f}, Tau={tau:.2f})")
                ax.set_xlabel("Calcium Signal")
                ax.set_ylabel("Derivative")
                ax.legend()
                plt.show()
                
            return slope
        except Exception as e:
            return 0.0
            
    return _remove_outliers_unified(signal, deriv, _dbscan_core, do_plot=do_plot)

def remove_outliers_zscore(signal, deriv, threshold=2, do_plot=False):
    def _zscore_core(x, y, threshold=2, do_plot=False):
        z_x = (x - np.mean(x)) / (np.std(x) + 1e-9)
        z_y = (y - np.mean(y)) / (np.std(y) + 1e-9)
        mask = (np.abs(z_x) < threshold) & (np.abs(z_y) < threshold)
        inlier_x = x[mask]
        inlier_y = y[mask]
        if len(inlier_x) < 2: return 0.0
        s, _ = np.polyfit(inlier_x, inlier_y, 1)
        return s
    return _remove_outliers_unified(signal, deriv, _zscore_core, threshold=threshold, do_plot=do_plot)

def scatter_plot(signal, deriv, title='Signal vs Derivative'):
    fig, ax = plt.subplots()
    ax.scatter(signal, deriv, marker='.', s=5)
    ax.set_title(title)

def scatter_all(signal, window_length=5):
    # Just a plotting helper, simple handling
    # If 2D, we flatten to show global distribution
    from scipy.signal import savgol_filter
    s = np.array(signal)
    smooth_cal = savgol_filter(s, window_length=window_length, deriv=0, delta=1., polyorder=3, axis=-1)
    smooth_deriv = savgol_filter(s, window_length=window_length, deriv=1, delta=1., polyorder=3, axis=-1)
    scatter_plot(smooth_cal.flatten(), smooth_deriv.flatten())
