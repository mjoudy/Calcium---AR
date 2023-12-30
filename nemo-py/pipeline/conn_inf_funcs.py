import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig
#import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

import remove_outliers as ro

plt.style.use('ggplot')
#plt.style.use('seaborn')

#sns.set_style('white')

def time_const(cut_signal, cut_deriv):
    
    tau_est.append(ro.pure_fit(cut_signal, cut_deriv))

    return np.array(tau_est)


def reconstructed_spikes(signal, deriv, tau_est):
    #shall be upgraded for using different methods of outlier removal
    #return np.array(deriv + (-tau_est)*signal)
    return (deriv + (-tau_est)*signal)

'''
def reconstructed_spikes(signal, deriv, cut_signal, cut_deriv):
    #shall be upgraded for using different methods of outlier removal
    tau_est = ro.pure_fit(cut_signal, cut_deriv)
    
    return deriv + (-tau_est)*signal
'''


def conn_inf_LR(conn_matrix, signals_matrix, lag=10, save_mat=False):

    G = np.load(conn_matrix)
    G = G - (np.diag(np.diag(G)))
    print("shape G:", G.shape)

    signals = np.load(signals_matrix)
    print('shape signal:', signals.shape)
    Y = signals[:, lag:]
    print("shape Y:", Y.shape)
    Y_prime = signals[:, :-lag]
    print('shape Y_prime:', Y_prime.shape)

    yk = Y.T
    print('shape yk:', yk.shape)
    y_k = Y_prime.T
    print('shape Y_prime:', Y_prime.shape)

    reg = LinearRegression(n_jobs=-1).fit(y_k, yk)
    A = reg.coef_
    A = A - (np.diag(np.diag(A)))

    if save_mat:
       est_conn_name = f"{conn_matrix}_{signals}_{lag}.npy"
       np.save(est_conn_name, A)

    #corr_G_A = np.corrcoef(G.flatten(), A.flatten())[0, 1]

    return A, G
