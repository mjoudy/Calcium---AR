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

def make_calcium(input_file, tau=100):
    spikes_add =  input_file
    spikes = np.load(spikes_add)
    spikes_name = spikes_add.split('/')[-1]
    print('spikes length: ', spikes_name)
    print("loaded the spikes: ", dtm.now())

    calcium_signal, spikes = kef.sim_calcium(spikes, tau=tau, neuron_id=-1)
    print("simulated the calcium: ", dtm.now())
    print('calcium signal shape: ', calcium_signal.shape)

    #parts = spikes_name.split('-')
    #new_file_name = f"calcium-{tau}-{parts[1]}-{parts[2]}"
    new_file_name = spikes_name.replace('spikes', f'calcium_tau{calcium_tau}')

    calcium_save_path = os.path.join(calcium_dir, new_file_name)

    np.save(calcium_save_path, calcium_signal)


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: python3 preprocess.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    make_calcium(input_file, tau=calcium_tau)
    print("calcium signal of chunks simulated: ", dtm.now())
