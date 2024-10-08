import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig

import kernel_est_funcs as kef
import conn_inf_funcs as cif
import remove_outliers as ro
import kernel_fit as kf

import os
import sys
import time

plt.style.use('ggplot')

print("start: ", time.time())

def process_data(input_file):
    calcium_add =  input_file
    calcium_signal = np.load(calcium_add)
    calcium_name = calcium_add.split('/')[-1]
    print('calcium length: ', calcium_name)
    print("loaded the calcium: ", time.time())

    signal, deriv = kef.smoothed_signals(calcium_signal, 51, do_plots=False)
    print("smoothed signals and calculated the derivative by sav-gol: ", time.time())

    n_rows = np.shape(calcium_signal)[0]

    #this not an array, it is a list. what would be if it was an array?
    # Initialize an empty array to store the results
    signal_cut = []
    deriv_cut = []

    # Apply the function to each row
    for i in range(n_rows):
        signal_temp, deriv_temp = kef.cut_spikes(calcium_signal[i, :], signal[i, :], deriv[i, :], win_len=5)
        signal_cut.append(signal_temp)
        deriv_cut.append(deriv_temp)

    print("cut signal and the derivative at spikes occuring: ", time.time())
    
    n_rows = np.shape(calcium_signal)[0]
    tau_est = np.empty(n_rows)

    for i in range(n_rows):
        #tau_est[i] = kef.estimate_tau(signal_cut[i], deriv_cut[i], do_plots=False)
        tau_est[i] = kef.pure_fit(signal_cut[i], deriv_cut[i], do_plots=False)

    print("estimated tau: ", time.time())


    feed_signals = np.empty((n_rows, np.shape(calcium_signal)[1]))
    for i in range(n_rows):
        feed_signals[i] = cif.reconstructed_spikes(signal[i], deriv[i], tau_est[i])

    print("reconstructed spikes: ", time.time())

    parts = calcium_name.split('-')
    new_file_name = f"feed-{parts[1]}-{parts[2]}"

    save_dir = "/work/ws/nemo/fr_mj200-lasso_reg-0/pipeline/source_data/t-60e6/chunked-processed"
    feed_save_path = os.path.join(save_dir, new_file_name)

    np.save(feed_save_path, feed_signals)


#if __name__ == "__main__":: This line checks whether the script is being run as the main program.
# When a Python script is executed, the special variable __name__ is set to "__main__" if the script is
# the main program being run. If the script is imported as a module into another script, __name__ is
# set to the name of the script or module.
#if len(sys.argv) != 2:: Checks if the correct number of command-line arguments is provided.
# In this case, the script expects one argument, which is the path to the input file.
#sys.argv is a list in Python that contains the command-line arguments passed to the script.
# The first element, sys.argv[0], is the script's name itself. The subsequent elements, 
# starting from sys.argv[1], are the arguments provided by the user.len(sys.argv) gives the total number
#  of elements in sys.argv, which corresponds to the number of command-line arguments.
if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: python3 preprocess.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    process_data(input_file)
    print("feed signal of chunks created: ", time.time())