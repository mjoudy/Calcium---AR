import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig
import seaborn as sns

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


def dask_pre_process(signal, spikes, sg_win=31, win_len=5):
    # Check if `signal` and `spikes` have the same number of columns
    if signal.shape[1] != spikes.shape[1]:
        spikes = spikes[:, -signal.shape[1]:]  # Cut columns from the beginning of spikes if needed

    # Initialize arrays
    num_rows, num_cols = signal.shape
    feed = np.zeros((num_rows, num_cols))
    b_pure_fits = np.zeros(num_rows)

    # Process each row separately
    for row in range(num_rows):
        # Smooth the signal and its derivative for the current row
        smooth_cal = sig.savgol_filter(signal[row], window_length=sg_win, deriv=0, delta=1, polyorder=3)
        smooth_deriv = sig.savgol_filter(signal[row], window_length=sg_win, deriv=1, delta=1, polyorder=3)
        
        # Identify spikes and the surrounding indices to remove
        bool_check = np.all((spikes[row] == 0) | (spikes[row] == 1))
        if bool_check:
            event_spikes = np.where(spikes[row])[0]
        else:
            event_spikes = spikes[row].astype(int)

        remove_index = []
        for i in event_spikes:
            remove_index.extend(np.arange(i - win_len, i + win_len))
        
        remove_index = np.array(remove_index)
        remove_index = remove_index[(remove_index >= 0) & (remove_index < num_cols)]

        # Remove indices from smooth_cal and smooth_deriv for fitting purposes
        smooth_cal_nosp = np.delete(smooth_cal, remove_index)
        smooth_deriv_nosp = np.delete(smooth_deriv, remove_index)

        # Fit a line to the modified arrays and store b_pure_fit for the row
        if smooth_cal_nosp.size > 1:  # Ensure there's enough data for fitting
            b_pure_fit, _ = np.polyfit(smooth_cal_nosp, smooth_deriv_nosp, deg=1)
            b_pure_fits[row] = 1/b_pure_fit
        else:
            b_pure_fits[row] = 0  # Default value if fitting is not possible

        # Calculate feed for this row
        feed[row, :] = -b_pure_fits[row] * smooth_cal + smooth_deriv

    return feed, b_pure_fits


def dask_feed_raper(signal, spikes, win_len=5, sg_delta=31):
    feed, _ = dask_pre_process(signal, spikes, win_len, sg_delta)
    return feed

def dask_fits_raper(signal, spikes, win_len=5, sg_delta=31):
    _, b_pure_fits = dask_pre_process(signal, spikes, win_len, sg_delta)
    return b_pure_fits

