
import numpy as np
from sklearn.linear_model import LinearRegression
from .outliers import fit_linear, remove_outliers_ransac, remove_outliers_iqr, remove_outliers_zscore, remove_outliers_dbscan
from . import signal_utils
import matplotlib.pyplot as plt

def estimate_time_constant(cut_signal, cut_deriv):
    """
    Estimate time constant tau from signal and derivative.
    Assumes spikes are already cut out!
    """
    slope = fit_linear(cut_signal, cut_deriv)
    return slope


def estimate_tau_robust(signal, window_length=5, method='ransac', do_plot=False):
    """
    Estimate Tau directly from raw calcium signals without explicit spike cutting.
    Uses robust regression on the Phase Space (Signal vs Derivative) to identify the 
    decay dynamics (dense region) and ignore spike trajectories (loops/outliers).
    
    Args:
        signal (np.ndarray): 1D or 2D calcium signal.
        window_length (int): Smoothing window for derivative calculation.
        method (str): 'ransac' (default), 'iqr', or 'zscore'.
        do_plot (bool): Whether to plot the phase space and fit (for the first neuron).
        
    Returns:
        float or np.ndarray: Estimated Tau (ms). Returns -1/slope.
    """
    # 1. Calculate Smooth Signal and Derivative
    smooth_cal, smooth_deriv = signal_utils.get_signal_derivative_pair(signal, window_length=window_length)
    
    # 2. Fit Phase Space using Robust Regression
    if method == 'ransac':
        slopes = remove_outliers_ransac(smooth_cal, smooth_deriv, do_plot=do_plot)
    elif method == 'dbscan':
        slopes = remove_outliers_dbscan(smooth_cal, smooth_deriv, do_plot=do_plot)
    elif method == 'iqr':
        slopes = remove_outliers_iqr(smooth_cal, smooth_deriv, do_plot=do_plot)
    elif method == 'zscore':
        slopes = remove_outliers_zscore(smooth_cal, smooth_deriv, do_plot=do_plot)
    else:
        raise ValueError(f"Unknown method: {method}")
        
    # 3. Convert Slope to Tau
    # Slope (b) approx -1/tau -> Tau = -1/b
    
    # Handle scalar vs array return
    if np.ndim(slopes) == 0:
        if abs(slopes) < 1e-9: return 0.0
        return -1.0 / slopes
    else:
        taus = np.zeros_like(slopes)
        mask = np.abs(slopes) > 1e-9
        taus[mask] = -1.0 / slopes[mask]
        return taus


def reconstruct_spikes(signal, deriv, tau_est):
    """
    Reconstruct spikes from signal, derivative, and estimated tau parameter.
    
    Args:
        signal: Calcium signal
        deriv: Derivative
        tau_est: Parameter from fit (slope = -1/tau usually)
    
    Returns:
        Estimated spikes (feed)
    """
    # Original: return (deriv + (-tau_est)*signal)
    # If tau_est is the slope (b_pure_fit), this means:
    # deriv - b_pure_fit * signal.
    return (deriv + (-tau_est) * signal)


def infer_connectivity_lr(conn_matrix_path, signals, lag=10):
    """
    Infer connectivity using Linear Regression and compare with ground truth.
    
    Args:
        conn_matrix_path (str): Path to .npy file containing ground truth connectivity matrix.
        signals (np.ndarray): Signals array (N_neurons x Time).
        lag (int): Lag for regression.
        
    Returns:
        float: Correlation between ground truth and inferred connectivity.
    """
    G = np.load(conn_matrix_path)
    G = G - (np.diag(np.diag(G)))

    Y = signals[:, lag:]
    Y_prime = signals[:, :-lag]

    yk = Y.T # (Time-lag, N)
    y_k = Y_prime.T # (Time-lag, N)
    
    # Fit yk = y_k * A
    reg = LinearRegression(n_jobs=-1).fit(y_k, yk)
    A = reg.coef_ # (N, N)
    A = A - (np.diag(np.diag(A)))
    
    corr_G_A = np.corrcoef(G.flatten(), A.flatten())[0, 1]

    return corr_G_A
