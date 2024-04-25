import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig

import kernel_est_funcs as kef
import conn_inf_funcs as cif
import remove_outliers as ro
import kernel_fit as kf
from config import *

import os
import sys
from datetime import datetime as dtm

plt.style.use('ggplot')

print("start: ", dtm.now())

def process_data(input_file, sg_win_len, sg_pl_order, sg_delta, cut_win_len, save_dir, do_plot=False):
    print(input_file)
    calcium_add =  input_file
    calcium_signal = np.load(calcium_add)
    calcium_name = calcium_add.split('/')[-1]
    print(calcium_name)
    print('calcium length: ', calcium_name)
    print("loaded the calcium: ", dtm.now())

    signal, deriv = kef.smoothed_signals(calcium_signal, win_len=sg_win_len, pl_order=sg_pl_order, dlt=sg_delta, do_plots=do_plot)
    print("smoothed signals and calculated the derivative by sav-gol: ", dtm.now())

    n_rows = np.shape(calcium_signal)[0]

    #this not an array, it is a list. what would be if it was an array?
    # Initialize an empty array to store the results
    signal_cut = []
    deriv_cut = []

    # Apply the function to each row
    for i in range(n_rows):
        signal_temp, deriv_temp = kef.cut_spikes(calcium_signal[i, :], signal[i, :], deriv[i, :], win_len=cut_win_len)
        signal_cut.append(signal_temp)
        deriv_cut.append(deriv_temp)

    print("cut signal and the derivative at spikes occuring: ", dtm.now())

    n_rows = np.shape(calcium_signal)[0]
    tau_est = np.empty(n_rows)

    for i in range(n_rows):
        tau_est[i] = kef.pure_fit(signal_cut[i], deriv_cut[i], do_plot=do_plots)
        print("tau estimated: ", dtm.now())


    feed_signals = np.empty((n_rows, np.shape(calcium_signal)[1]))
    for i in range(n_rows):
        feed_signals[i] = cif.reconstructed_spikes(signal[i], deriv[i], tau_est[i])

    print("reconstructed spikes: ", dtm.now())
#name changer function is defined in config.py
    #feed_data_name = name_changer(calcium_name)
    feed_data_name = calcium_name.replace('calcium', 'feed')
    feed_save_path = os.path.join(save_dir, feed_data_name)
    print(feed_save_path)
    np.save(feed_save_path, feed_signals)

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: python3 preprocess.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    process_data(input_file, sg_win_len, sg_pl_order, sg_delta, cut_win_len, processed_dir)
    print("feed signal of chunks created: ", dtm.now())

