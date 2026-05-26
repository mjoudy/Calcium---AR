
import numpy as np

def simulate_calcium_signals(spikes, tau=100.0, dt=1.0, noise_std_intra=0.01, noise_std_extra=1.0, warmup_time=0):
    """
    Simulate calcium fluorescence traces from spike trains using vectorized operations.
    Treats input as a batch of signals (N x T). 1D inputs are treated as (1 x T).
    """
    spikes = np.array(spikes)
    input_ndim = spikes.ndim
    
    # 1. Standardize to 2D: (N_neurons, Time)
    # np.atleast_2d converts (T,) to (1, T) correctly for row vectors behavior
    spikes_proc = np.atleast_2d(spikes)
    
    # 2. Warmup Slicing
    if warmup_time > 0 and warmup_time < spikes_proc.shape[1]:
        spikes_proc = spikes_proc[:, warmup_time:]
    
    N, sim_dur = spikes_proc.shape
    
    # 3. Vectorized Simulation
    # Generate noise for all neurons at once
    noise_intra = np.random.normal(0, noise_std_intra, (N, sim_dur))
    spikes_noisy = spikes_proc + noise_intra
    
    calcium_nsp = np.zeros((N, sim_dur))
    const_A = np.exp((-1/tau)*dt)
    
    # Initial Condition
    if sim_dur > 0:
        calcium_nsp[:, 0] = spikes_proc[:, 0]
    
    # Time loop (Vectorized over N neurons)
    for t in range(1, sim_dur):
        calcium_nsp[:, t] = const_A * calcium_nsp[:, t-1] + spikes_noisy[:, t]
        
    noise_recording = np.random.normal(0, noise_std_extra, (N, sim_dur))
    calcium_nsp_noisy = calcium_nsp + noise_recording
    
    # 4. Restore dimensionality for 1D inputs
    if input_ndim == 1:
        return calcium_nsp_noisy[0]
    
    return calcium_nsp_noisy
