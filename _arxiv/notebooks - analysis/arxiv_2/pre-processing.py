
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig
import seaborn as sns

from sklearn.linear_model import LinearRegression

import functions.kernel_est_funcs as kef
import functions.conn_inf_funcs as cif
# import functions.remove_outliers as ro # This is not needed anymore
import functions.kernel_fit as kf

def plot_cumsum_slope(spikes, reconstructed_spikes, neuron_index):
    """
    Plots the cumulative sum of spikes vs. reconstructed spikes and calculates the slope.
    """
    cum_spikes = np.cumsum(spikes[neuron_index, :])
    cum_reconstructed = np.cumsum(reconstructed_spikes)
    
    fit = np.polyfit(cum_spikes, cum_reconstructed, 1)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(cum_spikes, cum_reconstructed, s=1, label='Data')
    plt.plot(cum_spikes, fit[0] * cum_spikes + fit[1], 'r', label=f'Fit (slope={fit[0]:.4f})')
    plt.title(f'Cumulative Sum Plot for Neuron {neuron_index}')
    plt.xlabel('Cumulative Sum of Original Spikes')
    plt.ylabel('Cumulative Sum of Reconstructed Spikes')
    plt.savefig('figures/cumsum_tau_100.png')
    plt.close()
    return fit[0]

def plot_binned_signals(spikes, reconstructed_spikes, neuron_index, bin_size=100):
    """
    Plots the binned original and reconstructed spike trains.
    """
    n_bins = int(len(spikes[neuron_index, :]) / bin_size)
    
    binned_spike = np.zeros(n_bins)
    binned_reconstructed = np.zeros(n_bins)

    c = 0
    for i in range(n_bins):
        binned_spike[i] = np.sum(spikes[neuron_index, c:c + bin_size])
        binned_reconstructed[i] = np.sum(reconstructed_spikes[c:c + bin_size])
        c += bin_size
        
    plt.figure(figsize=(20, 8))
    plt.plot(binned_spike, label='Original Binned Spikes')
    plt.plot(binned_reconstructed, label='Reconstructed Binned Spikes')
    plt.title(f'Binned Spike Comparison for Neuron {neuron_index} (bin size={bin_size})')
    plt.xlabel('Bin')
    plt.ylabel('Spike Count')
    plt.legend()
    plt.savefig('figures/binned_rate_tau_100.png')
    plt.close()

def plot_phase_space(signal, deriv, title_suffix=''):
    """
    Generates a phase space plot of the signal vs its derivative.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(signal, deriv, s=1)
    plt.title(f'Phase Space Plot {title_suffix}')
    plt.xlabel('Smoothed Calcium Signal')
    plt.ylabel('Derivative of Smoothed Signal')
    plt.grid(True)
    plt.savefig(f'figures/phase_plane_tau_100.png')
    plt.close()


def main():
    """
    Main function to run the pre-processing and analysis pipeline.
    """
    plt.style.use('seaborn-v0_8')

    # --- Data Loading and Simulation ---
    spikes = np.load("spikes-10e4-ms.npy")
    N, sim_dur_total = spikes.shape
    wup_time = 1000
    spikes = spikes[:, wup_time:]
    sim_dur = spikes.shape[1]

    # Simulate calcium signal with two-fold noise
    noise_intra = np.random.normal(0, 0.01, (N, sim_dur))
    spikes_noisy = spikes + noise_intra

    tau = 100
    calcium_nsp = np.zeros((N, sim_dur))
    const_A = np.exp((-1/tau) * 1)
    for t in range(1, sim_dur):
        calcium_nsp[:, t] = const_A * calcium_nsp[:, t-1] + spikes_noisy[:, t]
    
    noise_recording = np.random.normal(0, 1, (N, sim_dur))
    calcium_nsp_noisy = calcium_nsp + noise_recording
    
    n = 500  # Neuron to analyze

    # --- Denoising and Signal Processing ---
    # Using Savitzky-Golay filter for smoothing and differentiation
    smooth_cal = sig.savgol_filter(calcium_nsp_noisy[n, :], window_length=11, polyorder=3, deriv=0, delta=1.0)
    smooth_deriv = sig.savgol_filter(calcium_nsp_noisy[n, :], window_length=11, polyorder=3, deriv=1, delta=1.0)
    
    # Reconstruct spike signal based on the differential equation
    x_ncsp = smooth_deriv + (1/tau) * smooth_cal

    # --- Plotting Initial Signals ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    ax1.set_title('Simulated Dynamics: Noisy Calcium and Components')
    ax1.plot(calcium_nsp_noisy[n, :], label='Noisy Calcium Signal (nsp_noisy)')
    ax1.plot(calcium_nsp[n, :], label='Calcium from Noisy Spikes (nsp)')
    ax1.plot(spikes_noisy[n, :], label='Noisy Spikes')
    ax1.legend()
    
    ax2.set_title('Reconstructed vs Original Spikes')
    ax2.plot(x_ncsp, label='Reconstructed Spikes (from nsp_noisy)')
    ax2.plot(spikes[n, :], label='Original Spikes', alpha=0.7)
    ax2.legend()
    plt.xlabel('Time (ms)')
    fig.tight_layout()
    plt.savefig('figures/simulated_dynamics_tau_100.png')
    plt.close()

    # --- Analysis and Plotting Functions ---
    plot_cumsum_slope(spikes, x_ncsp, n)
    plot_binned_signals(spikes, x_ncsp, n, bin_size=100)
    plot_phase_space(smooth_cal, smooth_deriv, f'for Neuron {n}')

    # --- Tau Estimation Comparison ---
    true_tau = 100
    
    # Method 1: Cutting spike windows
    # We need the full smoothed signals for all neurons for this
    full_smooth_cal, full_smooth_deriv = kef.smoothed_signals(calcium_nsp_noisy, 51, do_plots=False)
    
    signal_cut_list = []
    deriv_cut_list = []
    for i in range(N):
         sc, dc = kef.cut_spikes(spikes[i,:], full_smooth_cal[i,:], full_smooth_deriv[i,:], win_len=5)
         signal_cut_list.append(sc)
         deriv_cut_list.append(dc)
    
    # Estimate tau for neuron 'n' using the windowed data
    slope_windowed = kef.pure_fit(signal_cut_list[n], deriv_cut_list[n], do_plot=False)
    estimated_tau_windowed = -1 / slope_windowed if slope_windowed != 0 else float('inf')

    # Method 2: Outlier removal
    slope_outlier = kef.iqr_outlier(smooth_cal, smooth_deriv, do_plot=False)
    estimated_tau_outlier = -1 / slope_outlier if slope_outlier != 0 else float('inf')

    print(f"--- Tau Estimation for Neuron {n} ---")
    print(f"True Tau: {true_tau}")
    print(f"Estimated Tau (Spike Windowing): {estimated_tau_windowed:.4f}")
    print(f"Estimated Tau (IQR Outlier Removal): {estimated_tau_outlier:.4f}")


if __name__ == '__main__':
    # Create figures directory if it doesn't exist
    import os
    if not os.path.exists('figures'):
        os.makedirs('figures')
    main()
