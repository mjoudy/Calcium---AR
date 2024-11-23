import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

import kernel_est_funcs as kef
import conn_inf_funcs as cif
import remove_outliers as ro
import kernel_fit as kf

#plt.style.use('ggplot')
plt.style.use('seaborn')


spikes = np.load('spikes-10e5-ms.npy')
calcium_signal, spikes = kef.sim_calcium(spikes, neuron_id=-1)
signal, deriv = kef.smoothed_signals(calcium_signal, 51, do_plots=False)

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

n_rows = np.shape(spikes)[0]
tau_est = np.empty(n_rows)
for i in range(n_rows):
    tau_est[i] = ro.pure_fit(signal_cut[i], deriv_cut[i], do_plot=False)

feed_signals = np.empty((n_rows, np.shape(spikes)[1]))
for i in range(n_rows):
    feed_signals[i] = cif.reconstructed_spikes(signal[i], deriv[i], tau_est[i])

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

cif.conn_inf_LR('connectivity-10e4-ms.npy', feed_signals)