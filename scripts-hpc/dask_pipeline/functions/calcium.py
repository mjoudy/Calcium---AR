import numpy as np
import h5py

'''
def sim_calcium(spikes_path, tau=100):

    with h5py.File(spikes_path, 'r') as h5_file:
        print("Available datasets:", list(h5_file.keys()))
        spikes = h5_file['spikes_trains'][:]

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

'''


def sim_calcium(spikes_path, tau=100):

    N, total_dur = spikes.shape
    wup_time = 100
    spikes = spikes[:, wup_time:]
    sim_dur = spikes.shape[1]

    dt = 1
    const_A = np.exp((-1 / tau) * dt)

    # Reuse the calcium array to reduce memory
    calcium_nsp_noisy = np.zeros((N, sim_dur))
    noise_intra = np.random.normal(0, 0.01, (N, sim_dur))
    spikes_noisy = spikes + noise_intra

    # Directly compute the calcium signal
    calcium_nsp_noisy[:, 0] = spikes_noisy[:, 0]

    for t in range(1, sim_dur):
        calcium_nsp_noisy[:, t] = const_A * calcium_nsp_noisy[:, t - 1] + spikes_noisy[:, t]

    # Add recording noise in-place
    noise_recording = np.random.normal(0, 1, calcium_nsp_noisy.shape)
    calcium_nsp_noisy += noise_recording

    return calcium_nsp_noisy

def sim_calcium_dask(spikes_chunk, tau, is_first_chunk=False, wup_time=100):
    """
    Compute calcium signals for a chunk of spikes.
    This function operates on smaller chunks of the data.
    """
    if is_first_chunk:
        # Remove warm-up period for the first chunk
        spikes_chunk = spikes_chunk[:, wup_time:]
    
    N, sim_dur = spikes_chunk.shape
    dt = 1
    const_A = np.exp((-1 / tau) * dt)

    # Initialize calcium signal for the chunk
    calcium_nsp_noisy = np.zeros((N, sim_dur))

    # Add noise
    noise_intra = np.random.normal(0, 0.01, (N, sim_dur))
    spikes_noisy = spikes_chunk + noise_intra

    # Compute calcium signal
    calcium_nsp_noisy[:, 0] = spikes_noisy[:, 0]
    del spikes_chunk, noise_intra
    for t in range(1, sim_dur):
        calcium_nsp_noisy[:, t] = (
            const_A * calcium_nsp_noisy[:, t - 1] + spikes_noisy[:, t]
        )

    # Add recording noise
    noise_recording = np.random.normal(0, 1, calcium_nsp_noisy.shape)
    calcium_nsp_noisy += noise_recording

    return calcium_nsp_noisy
