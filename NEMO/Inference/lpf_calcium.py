#this is a script to perform a low-pass filter on spike train in order to get 
#calcium signals with a certain time constant tau

import numpy as np
from datetime import datetime

spikes = np.load("spikes-60e6-ms.npy")

N = np.shape(spikes)[0]
wup_time = 1000
spikes = spikes[:, wup_time:]
sim_dur = np.shape(spikes)[1]
calcium = np.zeros((N, sim_dur))
tau = 10
dt = 1
const_A = np.exp((-1/tau)*dt)

calcium[:, 0] = spikes[:, 0]

print("started: ", datetime.now())

for t in range(1, sim_dur):
    #calcium[:, t] = np.dot(const_A, calcium[:, t-1]) + spikes[:, t]
    calcium[:, t] = const_A*calcium[:, t-1] + spikes[:, t]

print("ended: ", datetime.now())
    
#there is no need for convolutional for loops!

np.save("calcium-60e6-tau-1.npy", calcium)