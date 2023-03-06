import numpy as np

from datetime import datetime as dtm

spikes = np.load("spikes-10e5-ms.npy")

current_time = dtm.now()
time_str = current_time.strftime("%H:%M:%S")
with open('log.txt', 'a') as f:
    f.write('spikes loaded: ' + time_str + '\n')

N = np.shape(spikes)[0]
wup_time = 1000
spikes = spikes[:, wup_time:]
sim_dur = np.shape(spikes)[1]
t_len = int(sim_dur/2)

spikes_split = np.array_split(spikes, 2, axis=1)
spikes_1 = spikes_split[0]
spikes_2 = spikes_split[1]

del spikes
np.save('spikes_1.npy', spikes_1)
np.save('spikes_2.npy', spikes_2)
del spikes_1, spikes_2

noise_intra_1 = np.random.normal(0, 0.01, (N, t_len))
noise_intra_2 = np.random.normal(0, 0.01, (N, t_len))
np.save('noise_intra_1.npy', noise_intra_1)
np.save('noise_intra_2.npy', noise_intra_2)
del noise_intra_2, noise_intra_1

spikes_1 = np.load('spikes_1.npy')
noise_intra_1 = np.load('noise_intra_1.npy')
spikes_noisy_1 = spikes_1 + noise_intra_1
np.save('spikes_noisy_1.npy', spikes_noisy_1)
del spikes_noisy_1, spikes_1, noise_intra_1

spikes_2 = np.load('spikes_2.npy')
noise_intra_2 = np.load('noise_intra_2.npy')
spikes_noisy_2 = spikes_2 + noise_intra_2
np.save('spikes_noisy_2.npy', spikes_noisy_2)
del spikes_noisy_2, spikes_2, noise_intra_2

noise_recording_1 = np.random.normal(0,1, (N, t_len))
noise_recording_2 = np.random.normal(0,1, (N, t_len))
np.save('noise_recording_1.npy', noise_recording_1)
np.save('noise_recording_2.npy', noise_recording_2)
del noise_recording_1, noise_recording_2

spikes_noisy_1 = np.load('spikes_noisy_1.npy')

calcium_nsp_1 = np.zeros((N, t_len))
tau = 1000
dt = 1
const_A = np.exp((-1/tau)*dt)

calcium_nsp_1[:, 0] = spikes_noisy_1[:, 0]

for t in range(1, t_len):
    calcium_nsp_1[:, t] = const_A*calcium_nsp_1[:, t-1] + spikes_noisy_1[:, t]

noise_recording_1 = np.load('noise_recording_1.npy')
calcium_noisy_1 = calcium_nsp_1 + noise_recording_1
np.save('calcium_noisy_1.npy', calcium_noisy_1)
del calcium_noisy_1, noise_recording_1

spikes_noisy_2 = np.load('spikes_noisy_2.npy')

calcium_nsp_2 = np.zeros((N, t_len))
tau = 1000
dt = 1
const_A = np.exp((-1/tau)*dt)

calcium_nsp_2[:, 0] = spikes_noisy_2[:, 0]

for t in range(1, t_len):
    calcium_nsp_2[:, t] = const_A*calcium_nsp_2[:, t-1] + spikes_noisy_2[:, t]

noise_recording_2 = np.load('noise_recording_2.npy')
calcium_noisy_2 = calcium_nsp_2 + noise_recording_2
np.save('calcium_noisy_2.npy', calcium_noisy_2)
del calcium_noisy_2, noise_recording_2

calcium_noisy_1 = np.load('calcium_noisy_1.npy')
calcium_noisy_2 = np.load('calcium_noisy_2.npy')

calcium_noisy = calcium_noisy_1 + calcium_noisy_2
np.save('calcium_noisy.npy', calcium_noisy)