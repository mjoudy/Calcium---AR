import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig
#import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn import metrics

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

def find_overlap(data1, data2):
    min_max = max(min(data1), min(data2))
    max_min = min(max(data1), max(data2))
    if min_max < max_min:  # Ensure there is an overlap
        return min_max, max_min
    else:
        return None  # No overlap

def overlap_interval(G, A):
    G = G.flatten()
    A = A.flatten()
    A_exc = A[G>0]
    A_inh = A[G<0]
    A_unc = A[G==0]

    e_u_min, e_u_max = find_overlap(A_exc, A_unc)
    i_u_min, i_u_max = find_overlap(A_inh, A_unc)

    return e_u_min, e_u_max, i_u_min, i_u_max

def performance_score(thresholded_G, thresholded_A):
    confusion_matrix = metrics.confusion_matrix(thresholded_G, thresholded_A)

    score = 0
    for i in range(confusion_matrix.shape[0]):
        score += confusion_matrix[i, i]/sum(confusion_matrix[i, :])

    return score

