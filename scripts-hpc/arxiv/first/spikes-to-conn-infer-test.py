#part of the code is commented due to memory considerations. This part is 
#basically for comparison goals which is done in the notebooks for a shorter 
#time length data.

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from datetime import datetime as dtm

current_time = dtm.now()
time_str = current_time.strftime("%H:%M:%S")
with open('log.txt', 'w') as f:
    f.write('The code started: ' + time_str + '\n')


# load spikes train and simulate calcium signals as low-pass filter
#####################################
spikes = np.load("spikes-10e5-ms.npy")

current_time = dtm.now()
time_str = current_time.strftime("%H:%M:%S")
with open('log.txt', 'a') as f:
    f.write('spikes loaded: ' + time_str + '\n')

N = np.shape(spikes)[0]
wup_time = 1000
spikes = spikes[:, wup_time:]
sim_dur = np.shape(spikes)[1]

noise_intra = np.random.normal(0, 0.01, (N, sim_dur))
spikes_noisy = spikes + noise_intra

#calcium = np.zeros((N, sim_dur))
calcium_nsp = np.zeros((N, sim_dur))
tau = 1000
dt = 1
const_A = np.exp((-1/tau)*dt)

#calcium[:, 0] = spikes[:, 0]
calcium_nsp[:, 0] = spikes[:, 0]

'''
for t in range(1, sim_dur):
    calcium[:, t] = const_A*calcium[:, t-1] + spikes[:, t]
'''

for t in range(1, sim_dur):
    calcium_nsp[:, t] = const_A*calcium_nsp[:, t-1] + spikes_noisy[:, t]


noise_recording = np.random.normal(0,1, (N, sim_dur))
#calcium_noisy = calcium + noise_recording
calcium_nsp_noisy = calcium_nsp + noise_recording

current_time = dtm.now()
time_str = current_time.strftime("%H:%M:%S")
with open('log.txt', 'a') as f:
    f.write('Calcium simulated: ' + time_str + '\n')


del noise_recording, calcium_nsp, spikes_noisy, noise_intra, spikes  


#Savitzky-Golay filter
###################################
smooth_cal = sig.savgol_filter(calcium_nsp_noisy, window_length=31, deriv=0, delta=1., polyorder=3)
smooth_deriv = sig.savgol_filter(calcium_nsp_noisy, window_length=31, deriv=1, delta=1., polyorder=3)

x_ncsp = smooth_deriv + (1/tau)*smooth_cal

current_time = dtm.now()
time_str = current_time.strftime("%H:%M:%S")
with open('log.txt', 'a') as f:
    f.write('Noisy calcium deconvolved: ' + time_str + '\n')

del smooth_cal, smooth_deriv


#Connectivity Inference
####################################
G = np.load('connectivity-10e5-ms.npy')
G = G - (np.diag(np.diag(G)))

k = 10
Y = x_ncsp[:, k:]
Y_prime = x_ncsp[:, :-k]

yk = Y.T
y_k = Y_prime.T

reg = LinearRegression(n_jobs=-1).fit(y_k, yk)
a = reg.coef_
a = a - (np.diag(np.diag(a)))

corr_result = np.corrcoef(G.flatten(), a.flatten())[0, 1]

current_time = dtm.now()
time_str = current_time.strftime("%H:%M:%S")
with open('log.txt', 'a') as f:
    f.write('Noisy calcium deconvolved: ' + time_str + '\n')
    f.write('\n' + 'Correlation coefficient between ground trouth and estimated connectivity is: ' + str(corr_result))
