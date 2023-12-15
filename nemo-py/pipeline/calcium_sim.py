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


def make_calcium(input_file):
    spikes_add =  input_file
    spikes = np.load(spikes_add)
    spikes_name = spikes_add.split('/')[-1]
    print('spikes length: ', spikes_name)
    print("loaded the spikes: ", time.time())

    calcium_signal, spikes = kef.sim_calcium(spikes, tau=100, neuron_id=-1)
    print("simulated the calcium: ", time.time())
    print('calcium signal shape: ', calcium_signal.shape)

    parts = spikes_name.split('-')
    new_file_name = f"calcium-{parts[1]}-{parts[2]}"

    save_dir = "/work/ws/nemo/fr_mj200-lasso_reg-0/pipeline/source_data/t-60e6/chunked-calcium"
    calcium_save_path = os.path.join(save_dir, new_file_name)

    np.save(calcium_save_path, calcium_signal)


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: python3 preprocess.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    make_calcium(input_file)
    print("feed signal of chunks created: ", time.time())

