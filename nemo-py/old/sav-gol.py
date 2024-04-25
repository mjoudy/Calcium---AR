import numpy as np
import scipy as sp
import scipy.signal as sig

from datetime import datetime as dtm

current_time = dtm.now()
time_str = current_time.strftime("%H:%M:%S")
with open('sav-gol.txt', 'w') as f:
    f.write('The code started: ' + time_str + '\n')

signal = np.load('calcium_noisy_1.npy')

tau = 1000

#Savitzky-Golay filter
###################################
smooth_cal = sig.savgol_filter(signal, window_length=31, deriv=0, delta=1., polyorder=3)
smooth_deriv = sig.savgol_filter(signal, window_length=31, deriv=1, delta=1., polyorder=3)

x_ncsp = smooth_deriv + (1/tau)*smooth_cal

current_time = dtm.now()
time_str = current_time.strftime("%H:%M:%S")
with open('log.txt', 'a') as f:
    f.write('Noisy calcium deconvolved: ' + time_str + '\n')

np.save('deconvolved-calcium-1.npy', x_ncsp)
