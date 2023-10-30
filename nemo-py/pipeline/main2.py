import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig
#import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

import kernel_est_funcs as kef
import conn_inf_funcs as cif
import remove_outliers as ro
import kernel_fit as kf

import time

plt.style.use('ggplot')
#plt.style.use('seaborn')

#sns.set_style('white')

print("start: ", time.time())

# Load the data
spikes_add = '/home/fr/fr_fr/fr_mj200/spikes-10e5-ms.npy'
conn_mat_add = '/home/fr/fr_fr/fr_mj200/connectivity-10e5-ms.npy'
spikes = np.load(spikes_add)
print('spikes length: ', spikes_add)
print("loaded the spikes: ", time.time())

calcium_signal, spikes = kef.sim_calcium(spikes, neuron_id=-1)
print("simulated the calcium: ", time.time())

signal, deriv = kef.smoothed_signals(calcium_signal, 51, do_plots=False)
print("smoothed signals and calculated the derivative by sav-gol: ", time.time())


n_rows = np.shape(spikes)[0]

#this not an array, it is a list. what would be if it was an array?
# Initialize an empty array to store the results
signal_cut = []
deriv_cut = []

# Apply the function to each row
for i in range(n_rows):
    signal_temp, deriv_temp = kef.cut_spikes(spikes[i, :], signal[i, :], deriv[i, :], win_len=5)
    signal_cut.append(signal_temp)
    deriv_cut.append(deriv_temp)

print("cut signal and the derivative at spikes occuring: ", time.time())


n_rows = np.shape(spikes)[0]
tau_est = np.empty(n_rows)
for i in range(n_rows):
    tau_est[i] = ro.pure_fit(signal_cut[i], deriv_cut[i], do_plot=False)

print("estimated tau for each neuron: ", time.time())


feed_signals = np.empty((n_rows, np.shape(spikes)[1]))
for i in range(n_rows):
    feed_signals[i] = cif.reconstructed_spikes(signal[i], deriv[i], tau_est[i])

print("reconstructed spikes based on estimated taus: ", time.time())


'''
# define a function which combine the two blocks above
def feed_to_conn_inf(spikes, signal, deriv, win_len=5):
    n_rows = np.shape(spikes)[0]
    tau_est = np.empty(n_rows)
    for i in range(n_rows):
        tau_est[i] = ro.pure_fit(signal_cut[i], deriv_cut[i], do_plot=False)

    feed_signals = np.empty((n_rows, np.shape(spikes)[1]))
    for i in range(n_rows):
        feed_signals[i] = cif.reconstructed_spikes(signal[i], deriv[i], tau_est[i])

    return feed_signals

print("estimated tau for each neuron and reconstruct spikes: ", time.time())
'''

corr_reconst = cif.conn_inf_LR(conn_mat_add, feed_signals)
print('connectivity matrix estimated based on reconstructed spikes: ', time.time())

corr_spikes = cif.conn_inf_LR(conn_mat_add, spikes)
print('connectivity matrix estimated based on original spikes: ', time.time())

cor_signals = cif.conn_inf_LR(conn_mat_add, signal)
print('connectivity matrix estimated based on original signals: ', time.time())

print('correlation coefficient between estimated and ground truth connectivity matrix based on "reconstructed signals": ', corr_reconst)
print('correlation coefficient between estimated and ground truth connectivity matrix based on "original spikes": ', corr_spikes) 
print('correlation coefficient between estimated and ground truth connectivity matrix based on "original signals": ', cor_signals)  




