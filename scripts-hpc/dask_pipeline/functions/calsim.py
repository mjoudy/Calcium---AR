import numpy as np


def sim_calcium(spikes, tau=100):

    N, total_dur = spikes.shape
    wup_time = 100
    #spikes = spikes[:, wup_time:]
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

