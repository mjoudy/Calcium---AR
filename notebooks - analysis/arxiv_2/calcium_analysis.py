import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import os

# Try importing custom modules if they exist in the same directory
try:
    import kernel_est_funcs as kef
    import conn_inf_funcs as cif
    import remove_outliers as ro
    import kernel_fit as kf
except ImportError:
    pass 

def simulate_calcium(spikes, tau, dt):
    """
    Simulates calcium dynamics using an AR(1) model (exponential decay).
    Vectorized using scipy.signal.lfilter.
    """
    const_A = np.exp((-1/tau)*dt)
    
    # Filter coefficients for y[n] = A*y[n-1] + x[n]
    # Rearranged: y[n] - A*y[n-1] = x[n]
    b = [1]
    a = [1, -const_A]
    
    # Apply filter along the time axis (axis 1)
    calcium = sig.lfilter(b, a, spikes, axis=1)
    
    return calcium

def bin_data(data, bin_size):
    """Bins a 1D array by summing values within each bin."""
    n_bins = data.shape[0] // bin_size
    # Truncate data to be a multiple of bin_size for reshaping
    data_trunc = data[:n_bins*bin_size]
    return data_trunc.reshape(n_bins, bin_size).sum(axis=1)

def estimate_time_constant(signal, deriv, dt=1):
    """
    Estimates the time constant tau from the signal and its derivative.
    Uses linear regression on the decay phases (where derivative < 0).
    """
    # Filter for decay phases (derivative < 0) to exclude spikes
    mask = deriv < 0
    y_decay = deriv[mask]
    x_decay = signal[mask]
    
    if len(x_decay) == 0:
        return np.nan

    # Fit y' = m * y + c  =>  slope m should be -1/tau
    slope, intercept = np.polyfit(x_decay, y_decay, 1)
    
    estimated_tau = -1 / slope * dt
    return estimated_tau

def analyze_cumulative(original, reconstructed):
    """Calculates cumulative sums and the slope of their relationship."""
    cum_orig = np.cumsum(original)
    cum_recon = np.cumsum(reconstructed)
    slope, intercept = np.polyfit(cum_orig, cum_recon, 1)
    return cum_orig, cum_recon, slope

def main():
    # --- Parameters ---
    TAU = 100
    DT = 1
    WUP_TIME = 1000
    NOISE_INTRA_STD = 0.01
    NOISE_REC_STD = 1.0
    SAVGOL_WIN_CAL = 31
    SAVGOL_WIN_NSP = 11
    SAVGOL_POLY = 3
    BIN_SIZE = 100
    NEURON_IDX = 500
    DATA_FILE = "spikes-10e4-ms.npy"

    # --- Data Loading ---
    if not os.path.exists(DATA_FILE):
        print(f"File {DATA_FILE} not found. Creating dummy data for demonstration.")
        # Create dummy spikes (Poisson process)
        spikes = np.random.poisson(0.05, (1000, 10000)).astype(float)
    else:
        spikes = np.load(DATA_FILE)

    N = spikes.shape[0]
    # Warmup cut
    spikes = spikes[:, WUP_TIME:]
    sim_dur = spikes.shape[1]

    print(f"Loaded spikes: shape {spikes.shape}")

    # --- Simulation ---
    # 1. Add intracellular noise to spikes
    noise_intra = np.random.normal(0, NOISE_INTRA_STD, (N, sim_dur))
    spikes_noisy = spikes + noise_intra

    # 2. Generate calcium signals
    print("Simulating calcium signals...")
    # Clean spikes -> Calcium
    calcium = simulate_calcium(spikes, TAU, DT)
    # Noisy spikes -> Calcium
    calcium_nsp = simulate_calcium(spikes_noisy, TAU, DT)

    # 3. Add recording noise
    noise_recording = np.random.normal(0, NOISE_REC_STD, (N, sim_dur))
    calcium_noisy = calcium + noise_recording
    calcium_nsp_noisy = calcium_nsp + noise_recording

    # --- Denoising & Reconstruction (Single Neuron) ---
    print(f"Analyzing neuron {NEURON_IDX}...")
    
    # Case: Noisy spikes -> Calcium + Noise
    smooth_cal_nsp = sig.savgol_filter(calcium_nsp_noisy[NEURON_IDX, :], window_length=SAVGOL_WIN_NSP, deriv=0, delta=1., polyorder=SAVGOL_POLY)
    smooth_deriv_nsp = sig.savgol_filter(calcium_nsp_noisy[NEURON_IDX, :], window_length=SAVGOL_WIN_NSP, deriv=1, delta=1., polyorder=SAVGOL_POLY)
    
    # --- Time Constant Estimation ---
    est_tau = estimate_time_constant(smooth_cal_nsp, smooth_deriv_nsp, DT)
    print(f"True Tau: {TAU}")
    print(f"Estimated Tau (from derivative method): {est_tau:.2f}")

    # Reconstruction equation: x(t) = y'(t) + (1/tau)y(t)
    x_ncsp = smooth_deriv_nsp + (1/TAU)*smooth_cal_nsp

    # --- Analysis ---
    binned_spike = bin_data(spikes[NEURON_IDX, :], BIN_SIZE)
    binned_xnsp = bin_data(x_ncsp, BIN_SIZE)
    
    # --- Cumulative Sum Analysis ---
    cum_orig, cum_recon, cum_slope = analyze_cumulative(spikes[NEURON_IDX, :], x_ncsp)
    print(f"Cumulative Sum Slope (Reconstructed vs Original): {cum_slope:.4f}")

    # --- Plotting ---
    try:
        plt.style.use('seaborn-v0_8')
    except:
        pass # Fallback to default style
        
    # Time vectors for alignment
    t_raw = np.arange(calcium_nsp_noisy.shape[1])
    t_binned = np.arange(len(binned_spike)) * BIN_SIZE

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    axes[0].plot(t_binned, binned_spike, label='Original Spikes (Binned)')
    axes[0].plot(t_binned, binned_xnsp, label='Reconstructed (Binned)', alpha=0.7)
    axes[0].set_title(f"Spike Reconstruction (Neuron {NEURON_IDX})")
    axes[0].legend()

    axes[1].plot(t_raw, calcium_nsp_noisy[NEURON_IDX, :], label='Noisy Calcium Signal', color='green', alpha=0.5)
    
    # Plot spike events below the calcium signal
    spike_times = np.where(spikes[NEURON_IDX, :] > 0)[0]
    y_min = np.min(calcium_nsp_noisy[NEURON_IDX, :])
    y_amp = np.ptp(calcium_nsp_noisy[NEURON_IDX, :])
    axes[1].vlines(spike_times, y_min - 0.2 * y_amp, y_min - 0.05 * y_amp, color='black', linewidth=0.5, label='Spike Events')

    axes[1].set_title("Calcium Signal")
    axes[1].legend()

    axes[2].plot(t_raw, cum_orig, label='Cumulative Original', color='black')
    axes[2].plot(t_raw, cum_recon, label='Cumulative Reconstructed', color='orange', linestyle='--')
    axes[2].set_title(f"Cumulative Sums (Slope: {cum_slope:.2f})")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("calcium_analysis_plot.png")
    print("Analysis complete. Plot saved to 'calcium_analysis_plot.png'")

if __name__ == "__main__":
    main()
