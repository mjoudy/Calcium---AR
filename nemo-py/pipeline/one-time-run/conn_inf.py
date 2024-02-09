import os
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
from loging_pipeline import log_process


def infere_connectivity(data_dir, conn_matrix, feed_data, lag=10, save_mat=True):
    corr_G_A = cif.conn_inf_LR(data_dir+conn_matrix, data_dir+feed_data, lag=10, save_mat=True)
    print('correlation coefficient between ground truth and estimated connectivity matrix:' + str(corr_G_A))


log_process('cor. coef. of GR and estimated connectivity calculated by LR')
