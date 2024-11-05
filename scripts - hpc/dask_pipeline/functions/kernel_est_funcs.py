# this script consists of functions to simulate calcium signals from spike trains,
#smooth them using savitzky-golay and 4 different outlier removal methods in order to 
#for a line to scatter plot of signal-derivative of signal.

import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig
import seaborn as sns

from sklearn.linear_model import RANSACRegressor

plt.style.use('ggplot')
#plt.style.use('seaborn')

#sns.set_style('white')

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

def dask_smooth(signal, win_len=5):
    smooth_cal = sig.savgol_filter(signal, window_length=win_len, deriv=0, delta=1., polyorder=3)
    smooth_deriv = sig.savgol_filter(signal, window_length=win_len, deriv=1, delta=1., polyorder=3)

    return smooth_cal, smooth_deriv

def dask_cut_spikes(spikes, signal, deriv, win_len=5):
    # Use np.all to check if all elements are either 0 or 1
    bool_check = np.all((spikes == 0) | (spikes == 1))

    if bool_check:
        event_spikes = np.where(spikes)[0]
    else:
        event_spikes = spikes.astype(int)

    remove_index = []
    for i in event_spikes:
        remove_index.append(np.arange(i - win_len, i + win_len))
    
    remove_index = np.array(remove_index).flatten()
    remove_index = remove_index[remove_index > 0]
    remove_index = remove_index[remove_index < len(signal)]

    signal = np.delete(signal, remove_index)
    deriv = np.delete(deriv, remove_index)

    return signal, deriv


def dask_smooth_and_cut_spikes(signal, spikes, win_len=5):
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
        smooth_cal = sig.savgol_filter(signal[row], window_length=win_len, deriv=0, delta=1., polyorder=3)
        smooth_deriv = sig.savgol_filter(signal[row], window_length=win_len, deriv=1, delta=1., polyorder=3)
        
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

    print(f"b_pure_fits: {b_pure_fits}")

    return feed, b_pure_fits

def dask_smooth_and_cut_spikes_feed(signal, spikes, win_len=5):
    feed, _ = dask_smooth_and_cut_spikes(signal, spikes, win_len)
    return feed

def dask_smooth_and_cut_spikes_b_pure_fits(signal, spikes, win_len=5):
    _, b_pure_fits = dask_smooth_and_cut_spikes(signal, spikes, win_len)
    return b_pure_fits



def sim_calcium(spikes, tau=100, neuron_id=500):

    if neuron_id == -1:
        N = np.shape(spikes)[0]
        wup_time = 1000
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
    else:
        wup_time = 1000
        spikes = spikes[neuron_id, wup_time:]
        sim_dur = np.shape(spikes)[0]

        noise_intra = np.random.normal(0, 0.01, sim_dur)
        spikes_noisy = spikes + noise_intra

        calcium = np.zeros(sim_dur)
        calcium_nsp = np.zeros(sim_dur)
        dt = 1
        const_A = np.exp((-1/tau)*dt)

        calcium[0] = spikes[0]
        calcium_nsp[0] = spikes[0]

        for t in range(1, sim_dur):
            calcium[t] = const_A*calcium[t-1] + spikes[t]

        for t in range(1, sim_dur):
            calcium_nsp[t] = const_A*calcium_nsp[t-1] + spikes_noisy[t]

        noise_recording = np.random.normal(0,1, sim_dur)
        calcium_noisy = calcium + noise_recording
        calcium_nsp_noisy = calcium_nsp + noise_recording

    #return calcium, calcium_noisy, calcium_nsp, calcium_nsp_noisy
    #return calcium_nsp_noisy, spikes
    return calcium_nsp_noisy


def smoothed_signals(signal, win_len, do_plots=False):
    smooth_cal = sig.savgol_filter(signal, window_length=win_len, deriv=0, delta=1., polyorder=3)
    smooth_deriv = sig.savgol_filter(signal, window_length=win_len, deriv=1, delta=1., polyorder=3)

    if (do_plots==True) & (signal.ndim!=1):
        neuron_id = 500
        fig, ax1 = plt.subplots(figsize=(20,8))
        ax1.plot(smooth_cal[neuron_id, :])
        ax1.plot(30*smooth_deriv[neuron_id, :])

    if (do_plots==True) & (signal.ndim==1):
        fig, ax1 = plt.subplots(figsize=(20,8))
        ax1.plot(smooth_cal)
        ax1.plot(30*smooth_deriv)

    return smooth_cal, smooth_deriv


def cut_spikes(spikes, signal, deriv, win_len=5):
    
    bool_check = all(element==0 or element==1 for element in spikes)

    if bool_check:
        event_spikes = np.where(spikes)[0]
    else:
        event_spikes = spikes.astype(int)

    remove_index = []
    for i in event_spikes:
        remove_index.append(np.arange(i-win_len, i+win_len))
        #add a line to include cut spikes
    
    remove_index = np.array(remove_index)
    remove_index = remove_index.flatten()
    remove_index = remove_index[remove_index>0]
    remove_index = remove_index[remove_index<len(signal)]

    signal = np.delete(signal, remove_index)
    deriv = np.delete(deriv, remove_index)

    return signal, deriv


'''
#following function was a try for developing cut_spikes for multiple neuros
def cut_spikes1(spikes, signal, deriv, win_len=5):
    
    bool_check = np.all((spikes == 0) | (spikes == 1))

    if bool_check:
        for i in spikes.shape[0]:
            spikes[i] = np.where(spikes[i, :])[0]
    else:
        spikes = spikes.astype(int)

    remove_index = []
    for i in spikes.shape[0]:
        for j in spikes[i, j]:
            remove_index.append(np.arange(j-win_len, j+win_len))
            # Add a line to include cut spikes
    remove_index = np.array(remove_index)
    remove_index = remove_index.flatten()
    remove_index = remove_index[remove_index > 0]

    for idx in remove_index:
        signal = np.delete(signal, idx)
        deriv = np.delete(deriv, idx)

    return signal, deriv
'''


##### outlier removal functions

def scatter_all(signal, win_len):
    smooth_cal = sig.savgol_filter(signal, window_length=win_len, deriv=0, delta=1., polyorder=3)
    smooth_deriv = sig.savgol_filter(signal, window_length=win_len, deriv=1, delta=1., polyorder=3)

    fig, ax = plt.subplots()
    ax.scatter(smooth_cal, smooth_deriv, marker='.', s=5)
    ax.set_xlabel('Calcium Signal')
    ax.set_ylabel('Derivative of Calcium Signal')


def pure_fit(signal, deriv, do_plot=False):

    b_pure_fit, a_pure_fit = np.polyfit(signal, deriv, deg=1)

    if do_plot==True:
        fig, ax = plt.subplots()
        ax.scatter(signal, deriv, marker='.', s=5)
        ax.plot(signal, a_pure_fit+b_pure_fit*signal, color='k')
        ax.set_xlabel('Calcium Signal')
        ax.set_ylabel('Derivative of Calcium Signal')
    
    return b_pure_fit


def iqr_outlier(signal, deriv, threshold=1.5, percentile_start=25, percentile_end=75,  do_plot=False):
    
    x = np.array(signal)
    y = np.array(deriv)

    residuals = y - np.polyval(np.polyfit(x, y, 1), x)
    quartile_1, quartile_3 = np.percentile(residuals, [percentile_start, percentile_end])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (threshold * iqr)
    upper_bound = quartile_3 + (threshold * iqr)
    mask = (residuals >= lower_bound) & (residuals <= upper_bound)

    inlier_x = x[mask]
    inlier_y = y[mask]

    b_iqr, a_iqr = np.polyfit(x[mask], y[mask], deg=1)
    
    if do_plot==True:
        i_mask = np.logical_not(mask)
        outlier_x = x[i_mask]
        outlier_y = y[i_mask]
        fig, ax = plt.subplots()
        ax.scatter(inlier_x, inlier_y, marker='.', s=5)
        ax.plot(inlier_x, a_iqr+b_iqr*inlier_x, color='k')
        ax.scatter(outlier_x, outlier_y, marker='.', s=5, color='red')
        ax.set_xlabel('Calcium Signal')
        ax.set_ylabel('Derivative of Calcium Signal')

    return b_iqr


def ransac_outlier(signal, deriv, do_plot=False):

    ransac = RANSACRegressor()
    x = np.array(signal).reshape(-1, 1)
    y = np.array(deriv)
    ransac.fit(x, y)

    if do_plot==True:
        mask = ransac.inlier_mask_
        i_mask = np.logical_not(mask)
        line_x = np.linspace(min(x), max(x), 100).reshape(-1, 1)
        line_y = ransac.predict(line_x)

        fig, ax = plt.subplots()
        plt.scatter(x[mask], y[mask], label='Inliers', marker='o', s=5)
        plt.scatter(x[i_mask], y[i_mask], label='Outliers', color='red', marker='o', s=5)
        plt.plot(line_x, line_y, color='green', label='Robust Regression Line')
        ax.set_xlabel('Calcium Signal')
        ax.set_ylabel('Derivative of Calcium Signal')

    
    return ransac.estimator_.coef_[0]


def zscore_outlier(signal, deriv, threshold=2, do_plot=False):

    x = np.array(signal)
    y = np.array(deriv)
    z_scores_x = (signal - np.mean(signal)) / np.std(signal)
    z_scores_y = (deriv - np.mean(deriv)) / np.std(deriv)
    mask = np.abs(z_scores_x) < threshold
    mask &= np.abs(z_scores_y) < threshold

    b_zscore, a_zscore = np.polyfit(x[mask], y[mask], deg=1)

    if do_plot==True:
        i_mask=np.logical_not(mask)
        fig, ax = plt.subplots()
        plt.scatter(x[mask], y[mask], marker='.', s=5)
        plt.scatter(x[i_mask], y[i_mask], marker='.', s=5,color='red')
        plt.plot(x[mask], a_zscore+b_zscore*x[mask], color='k')
        ax.set_xlabel('Calcium Signal')
        ax.set_ylabel('Derivative of Calcium Signal')

    return b_zscore



