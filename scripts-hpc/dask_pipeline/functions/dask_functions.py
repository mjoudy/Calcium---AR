import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig
import seaborn as sns
from functions import remove_outliers as ro

def dask_calcium(spikes, tau=100):

    N = np.shape(spikes)[0]
    wup_time = 100
    spikes = spikes[:, wup_time:]
    sim_dur = np.shape(spikes)[1]
    
    noise_intra = np.random.normal(0, 0.01, (N, sim_dur))
    spikes_noisy = spikes + noise_intra

    calcium = np.zeros((N, sim_dur))
    calcium_nsp = np.zeros((N, sim_dur))
    dt = 1
    const_A = np.exp((-1/tau)*dt)

    calcium[:, 0] = spikes[:, 0]
    calcium_nsp[:, 0] = spikes[:, 0]

    for t in range(1, sim_dur):
        calcium[:, t] = const_A*calcium[:, t-1] + spikes[:, t]

    for t in range(1, sim_dur):
        calcium_nsp[:, t] = const_A*calcium_nsp[:, t-1] + spikes_noisy[:, t]

    noise_recording = np.random.normal(0,1, (N, sim_dur))
    calcium_noisy = calcium + noise_recording
    calcium_nsp_noisy = calcium_nsp + noise_recording

    return calcium_nsp_noisy


def apply_sav_gol(signal_row, sg_win):
    smooth_cal = sig.savgol_filter(signal_row, window_length=sg_win, deriv=0, delta=1, polyorder=3)
    smooth_deriv = sig.savgol_filter(signal_row, window_length=sg_win, deriv=1, delta=1, polyorder=3)
    return smooth_cal, smooth_deriv

def calculate_mask(spikes_row, win_len, num_cols):
    """Calculate the mask to exclude indices around spikes."""
    mask = np.ones(num_cols, dtype=bool)
    event_spikes = np.where(spikes_row)[0]
    
    for i in event_spikes:
        start = max(0, i - win_len)
        end = min(num_cols, i + win_len)
        mask[start:end] = False
    
    return mask

def perform_polyfit(smooth_cal, smooth_deriv, mask):
    """Perform polynomial fitting on masked data."""
    #b is slope of the fitted line of signal-derivative. -1/b=tau.
    valid_indices = np.where(mask)[0]
    if valid_indices.size > 1:  # Ensure sufficient data for fitting
        #b_pure_fit, _ = np.polyfit(smooth_cal[valid_indices], smooth_deriv[valid_indices], deg=1)
        b_pure_fit = ro.pure_fit(smooth_cal[valid_indices], smooth_deriv[valid_indices])
        return b_pure_fit
    else:
        return 0  # Default if insufficient data

def calculate_feed(smooth_cal, smooth_deriv, b_pure_fit):
    return (-b_pure_fit)*smooth_cal + smooth_deriv

def dask_preprocess(signal, spikes, sg_win=31, win_len=5):

    if signal.shape[1] != spikes.shape[1]:
        spikes = spikes[:, -signal.shape[1]:]

    num_rows, num_cols = signal.shape
    feed = np.zeros((num_rows, num_cols))
    b_pure_fits = np.zeros(num_rows)

    for row in range(num_rows):
        smooth_cal, smooth_deriv = apply_sav_gol(signal[row], sg_win)
        mask = calculate_mask(spikes[row], win_len, num_cols)
        b_pure_fits[row] = perform_polyfit(smooth_cal, smooth_deriv, mask)
        feed[row, :] = calculate_feed(smooth_cal, smooth_deriv, b_pure_fits[row])

    return feed

def dask_estimate_kernels(signal, spikes, sg_win=31, win_len=5):

    if signal.shape[1] != spikes.shape[1]:
        spikes = spikes[:, -signal.shape[1]:]

    num_rows, num_cols = signal.shape
    b_pure_fits = np.zeros(num_rows)

    for row in range(num_rows):
        smooth_cal, smooth_deriv = apply_sav_gol(signal[row], sg_win)
        mask = calculate_mask(spikes[row], win_len, num_cols)
        b_pure_fits[row] = perform_polyfit(smooth_cal, smooth_deriv, mask)

    return b_pure_fits


