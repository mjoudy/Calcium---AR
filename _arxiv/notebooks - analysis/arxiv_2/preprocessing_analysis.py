import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import pandas as pd
from scipy.stats import pearsonr
import os

from functions import kernel_est_funcs as kef

# --- 1. CONFIGURATION & PARAMETERS ---
TAU_VALUES = [50, 100, 200, 300]  # Biological range (ms) [cite: 12]
SG_WINDOWS = [11, 31, 51]         # Smoothing scales 
BIN_SIZE = 50                    # Binning for rate preservation [cite: 95]
DT = 1
POLYORDER = 3

def simulate_dynamics(spikes, tau):
    """Simulates calcium signal with decay tau[cite: 17, 18]."""
    sim_dur = spikes.shape[1]
    cal = np.zeros_like(spikes)
    const_A = np.exp((-1/tau)*DT)
    for t in range(1, sim_dur):
        cal[:, t] = const_A * cal[:, t-1] + spikes[:, t]
    # Vectorized simulation using lfilter (y[n] = A*y[n-1] + x[n])
    b = [1]
    a = [1, -const_A]
    cal = sig.lfilter(b, a, spikes, axis=1)
    return cal

def deconvolve(cal_noisy, tau, window):
    """Applies Savitzky-Golay deconvolution[cite: 60, 61, 62]."""
    smooth_cal = sig.savgol_filter(cal_noisy, window_length=window, polyorder=POLYORDER, deriv=0)
    smooth_der = sig.savgol_filter(cal_noisy, window_length=window, polyorder=POLYORDER, deriv=1)
    # The proxy equation: x(t) = y_dot + (1/tau)y
    return smooth_der + (1/tau) * smooth_cal

# --- 2. MAIN ANALYSIS ENGINE ---
def run_full_analysis(file_path):
    # Load and Preprocess [cite: 3, 6]
    spikes_raw = np.load(file_path)
    spikes = spikes_raw[:20, 1000:] # Use subset of neurons and time for speed
    N, T = spikes.shape
    
    # Storage for "Mass Analysis" statistics
    stats = []

    for tau in TAU_VALUES:
        cal = simulate_dynamics(spikes, tau)
        # Add realistic recording noise [cite: 23, 24]
        noise = np.random.normal(0, 0.5, cal.shape)
        cal_noisy = cal + noise
        
        for win in SG_WINDOWS:
            recon_all = np.zeros_like(spikes)
            corrs = []
            slopes = []

            for n in range(N):
                recon = deconvolve(cal_noisy[n], tau, win)
                recon_all[n] = recon
                
                # Correlation of binned rates [cite: 101, 102]
                n_bins = T // BIN_SIZE
                b_spike = np.sum(spikes[n, :n_bins*BIN_SIZE].reshape(-1, BIN_SIZE), axis=1)
                b_recon = np.sum(recon[:n_bins*BIN_SIZE].reshape(-1, BIN_SIZE), axis=1)
                corrs.append(pearsonr(b_spike, b_recon)[0])

                # Cumsum slope calculation [cite: 83, 85, 87]
                fit = np.polyfit(np.cumsum(spikes[n]), np.cumsum(recon), 1)
                slopes.append(fit[0])

            stats.append({
                'tau': tau, 'win': win, 
                'mean_corr': np.mean(corrs), 
                'mean_slope': np.mean(slopes),
                'slope_err': np.std(slopes)
            })

            # --- PLOTTING (Example for one neuron, first tau/win pair) ---
            if tau == TAU_VALUES[1] and win == SG_WINDOWS[1]:
                generate_required_plots(spikes[0], cal_noisy[0], recon_all[0], b_spike, b_recon, tau)

    return stats

# --- 3. FIGURE GENERATION ---
def generate_required_plots(gt_spikes, cal_noisy, recon, b_spike, b_recon, tau):
    save_dir = "figures"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Figure 1: Simulated Signals [cite: 42, 43, 45, 48]
    plt.figure(figsize=(15, 4))
    plt.plot(cal_noisy, label='Noisy Calcium', alpha=0.7)
    # Plot spikes below the horizontal axis (negative)
    plt.plot(gt_spikes * -5, label='Spikes', color='black')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5) # Add a line at 0 for reference
    plt.title(f"Simulated Dynamics (Tau={tau}ms)")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"simulated_dynamics_tau_{tau}.png"))
    plt.close()

    # Figure 2: Binned Rate Preservation [cite: 114, 115, 116]
    plt.figure(figsize=(15, 4))
    plt.step(range(len(b_spike)), b_spike/b_spike.max(), label='GT Binned Rate')
    plt.step(range(len(b_recon)), b_recon/b_recon.max(), label='Recon Binned Rate', alpha=0.8)
    plt.title("Binned Rate Preservation (Normalized)")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"binned_rate_tau_{tau}.png"))
    plt.close()

    # Figure 3: Cumulative Sum (Cumsum) [cite: 89, 91] - UPDATED
    cum_spikes = np.cumsum(gt_spikes)
    cum_x_ncsp = np.cumsum(recon)

    fit1 = np.polyfit(cum_spikes, cum_x_ncsp, 1)
    # y = fit1[0]*cum_spikes + fit1[1]

    f = plt.figure(figsize=(5,3))
    ax = f.subplots()
    ax.scatter(cum_spikes, cum_x_ncsp, s=1)
    #ax.plot(cum_spikes, y)
    
    plt.xlabel("Cumulative Ground Truth Spikes")
    plt.ylabel("Cumulative Reconstructed Proxy")
    plt.title("Mass Preservation (Cumsum)")
    plt.savefig(os.path.join(save_dir, f"cumsum_tau_{tau}.png"))
    plt.close()

    # Figure 4: Phase Plane (Derivative vs Signal) [cite: 154, 171]
    # We use Savitzky-Golay for the y-axis derivative
    dy = sig.savgol_filter(cal_noisy, window_length=31, polyorder=3, deriv=1)
    plt.figure(figsize=(8, 6))
    plt.scatter(cal_noisy, dy, s=2, alpha=0.3)
    # Highlight the decay slope (-1/tau)
    x_range = np.linspace(0, cal_noisy.max(), 100)
    plt.plot(x_range, -1/tau * x_range, color='red', linestyle='--', label='Theoretical Decay')
    plt.xlabel("Calcium (y)")
    plt.ylabel("Derivative (dy/dt)")
    plt.title("Phase Plane Analysis")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"phase_plane_tau_{tau}.png"))
    plt.close()

# Run
# stats_summary = run_full_analysis('spikes-10e4-ms.npy')
if __name__ == "__main__":
    data_file = os.path.join(os.path.dirname(__file__), 'spikes-10e4-ms.npy')

    stats_summary = run_full_analysis(data_file)
    
    df_stats = pd.DataFrame(stats_summary)
    print(df_stats.to_string())