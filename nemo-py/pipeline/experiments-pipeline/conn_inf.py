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
from loging_pipeline import *
from config import *
#plt.style.use('ggplot')
#plt.style.use('seaborn')
#sns.set_style('white')

feed = data_dir + '/' + final_data_name
gr = data_dir + '/' + conn_matrix
corr_G_A = cif.conn_inf_LR(gr, feed, lag=inf_lag, save_mat=True)

add_results_to_json('Corr. Coef.', corr_G_A, 'experiments_log.json')

print('correlation coefficient between ground truth and estimated connectivity matrix:' + str(corr_G_A))
log_process('cor. coef. of GR and estimated connectivity calculated by LR')

append_dict_to_csv('experiments_log.json', 'experiments_log.csv')

os.remove('chunk_name.txt')
os.remove('cal_chunk_name.txt')
