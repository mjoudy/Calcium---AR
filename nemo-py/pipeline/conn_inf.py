import os
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
#plt.style.use('seaborn')

#sns.set_style('white')

source_dir = '/work/ws/nemo/fr_mj200-lasso_reg-0/pipeline/source_data/t-60e6'
gt_address = source_dir + '/'+'connectivity-10e4-ms.npy'
processed_signals_address = source_dir + '/'+'feed-60e6-msfinal_data.npy'

ground_truth = np.load(gt_address)
processed_signals = np.load(processed_signals_address)

estimated_conn, ground_truth = cif.conn_inf_LR(gt_address, processed_signals, lag=10)
#print('correlation coefficient between ground truth and estimated connectivity matrix:' + str(corr_G_A))

corr_G_A = np.corrcoef(ground_truth.flatten(), estimated_conn.flatten())[0, 1]

