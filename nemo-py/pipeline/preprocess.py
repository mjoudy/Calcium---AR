import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig
#import seaborn as sns

#from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import Lasso
#from sklearn.linear_model import Ridge

import kernel_est_funcs as kef
import conn_inf_funcs as cif
import remove_outliers as ro
import kernel_fit as kf

import time
#import cProfile
#from memory_profiler import profile

#profiler = cProfile.Profile()

plt.style.use('ggplot')
#plt.style.use('seaborn')

#sns.set_style('white')

print("start: ", time.time())


#profiler.enable()

def process_data(input_file):
    # Load the data
    spikes_add = input_file
    #conn_mat_add = 'connectivity-10e4-ms.npy'
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
        tau_est[i] = kef.estimate_tau(signal_cut[i], deriv_cut[i], do_plots=False)

    print("estimated tau: ", time.time())


    feed_signals = np.empty((n_rows, np.shape(spikes)[1]))
    for i in range(n_rows):
        feed_signals[i] = cif.reconstructed_spikes(signal[i], deriv[i], tau_est[i])

    print("reconstructed spikes: ", time.time())

    #the processed chunks are going to be overwritten in the same file
    np.save(input_file, feed_signals)


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


