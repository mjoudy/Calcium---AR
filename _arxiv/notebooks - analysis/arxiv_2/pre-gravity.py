
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig
import seaborn as sns
import sys
import os

# Add functions directory to path
sys.path.append(os.path.join(os.getcwd(), 'functions'))

import functions.kernel_est_funcs as kef
import functions.conn_inf_funcs as cif
import functions.remove_outliers as ro
import functions.kernel_fit as kf

def plot_calcium_spikes(time, calcium, spikes_times, title="Calcium Signal with Spike Events"):
    """
    Plots the simulated calcium signal with spike events as short vertical dashes.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time, calcium, label='Calcium Signal')
    
    # Plot spikes as vertical dashes below the signal
    # Assuming spike_times are indices where spikes occur
    ymin, ymax = plt.ylim()
    # Place ticks slightly below the minimum signal
    tick_min = ymin - (ymax - ymin) * 0.1
    tick_max = ymin - (ymax - ymin) * 0.05
    
    plt.vlines(spikes_times, tick_min, tick_max, colors='r', label='Spikes', linewidth=1)
    plt.title(title)
    plt.legend()
    plt.show()

def plot_cumsum_slope(signal, title="Cumsum of Signal"):
    """
    Plots the cumulative sum of the mean-subtracted signal and finds its slope.
    """
    cumsum_sig = np.cumsum(signal - np.mean(signal))
    
    # Fit a line to find the slope
    x = np.arange(len(cumsum_sig))
    slope, intercept = np.polyfit(x, cumsum_sig, 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, cumsum_sig, label='Cumsum')
    plt.plot(x, slope * x + intercept, 'r--', label=f'Fit (Slope={slope:.4f})')
    plt.title(title)
    plt.legend()
    plt.show()
    
    print(f"Slope of Cumsum ({title}): {slope}")

def plot_phase_space(signal, deriv, title="Phase Space Plot"):
    """
    Plots the phase space (Signal vs Derivative).
    """
    plt.figure(figsize=(8, 8))
    plt.plot(signal, deriv, '.', markersize=2)
    plt.xlabel('Signal')
    plt.ylabel('Derivative')
    plt.title(title)
    plt.show()

def plot_binned_signals(original_spikes_binned, reconstructed_spikes_binned, title="Binned Signals Comparison"):
    """
    Plots the binned original spikes vs reconstructed spikes.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(original_spikes_binned, label='Original Spikes (Binned)')
    plt.plot(reconstructed_spikes_binned, label='Reconstructed Spikes (Binned)', alpha=0.7)
    plt.title(title)
    plt.legend()
    plt.show()

def bin_signal(signal, bin_size=10):
    """
    Bins the signal by summing over bin_size windows.
    """
    return np.add.reduceat(signal, np.arange(0, len(signal), bin_size))

def main():
    # 1. Load Data
    print("Loading data...")
    try:
        spikes = np.load('spikes-10e4-ms.npy')
    except FileNotFoundError:
        print("File 'spikes-10e4-ms.npy' not found. Trying 'spikes-10e5-ms.npy'...")
        try:
            spikes = np.load('spikes-10e5-ms.npy')
        except FileNotFoundError:
            print("No spikes file found.")
            return

    # 2. Simulation
    # Use neuron_id=0 for single neuron analysis as implied by plotting requirements
    neuron_id = 0 
    print(f"Simulating calcium signal for neuron {neuron_id}...")
    
    # Manually slice spikes to match wup_time in sim_calcium
    wup_time = 1000
    if spikes.ndim == 2:
        spikes_sliced = spikes[neuron_id, wup_time:]
    else:
        spikes_sliced = spikes[wup_time:]
        
    # Simulate Calcium
    calcium_signal = kef.sim_calcium(spikes, tau=100, neuron_id=neuron_id)
    
    # 3. Smoothing
    print("Smoothing signal...")
    win_len = 51
    # For 1D array, smoothed_signals returns 1D arrays
    smooth_cal, smooth_deriv = kef.smoothed_signals(calcium_signal, win_len, do_plots=False)
    
    # 4. Plotting
    print("Generating plots...")
    
    # New Plot: Calcium with Spikes
    spike_indices = np.where(spikes_sliced > 0)[0]
    time = np.arange(len(calcium_signal))
    plot_calcium_spikes(time, calcium_signal, spike_indices, title="Simulated Calcium Signal with Spikes")
    
    # Cumsum Slope
    plot_cumsum_slope(calcium_signal, title="Cumsum of Calcium Signal")
    
    # Phase Space
    plot_phase_space(smooth_cal, smooth_deriv, title="Phase Space (Smoothed)")
    
    # Binned Signals (Original vs Reconstructed)
    # Reconstruct spikes using fixed Tau=100 (True Tau) for visual comparison
    true_tau = 100
    rec_spikes = cif.reconstructed_spikes(smooth_cal, smooth_deriv, true_tau)
    
    # Binning
    bin_size = 50
    spikes_binned = bin_signal(spikes_sliced, bin_size)
    rec_spikes_binned = bin_signal(rec_spikes, bin_size)
    
    plot_binned_signals(spikes_binned, rec_spikes_binned, title=f"Binned Spikes (Bin Size {bin_size}) - True Tau {true_tau}")
    
    # 5. Tau Estimation
    print("\nEstimating Tau...")
    
    # Method 1: Cut Spikes -> Fit
    print("Method 1: Cutting Spikes Window")
    # kef.cut_spikes returns the signal with spikes REMOVED
    # ro.pure_fit estimates the slope (-1/tau)
    sig_cut, deriv_cut = kef.cut_spikes(spikes_sliced, smooth_cal, smooth_deriv, win_len=5)
    slope_m1 = ro.pure_fit(sig_cut, deriv_cut, do_plot=False)
    tau_est_m1 = -1 / slope_m1 if slope_m1 != 0 else np.nan
    print(f"Method 1 Estimated Tau: {tau_est_m1:.2f}")
    
    # Method 2: Outlier Removal (All data points)
    print("Method 2: Outlier Removal form Phase Space")
    # Using iqr_outlier on the FULL smoothed signals
    slope_m2 = ro.iqr_outlier(smooth_cal, smooth_deriv, do_plot=False)
    tau_est_m2 = -1 / slope_m2 if slope_m2 != 0 else np.nan
    print(f"Method 2 Estimated Tau: {tau_est_m2:.2f}")
    
    print(f"True Tau: {true_tau}")

if __name__ == "__main__":
    main()
