import numpy as np
import scipy as sp

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from datetime import datetime as dtm

current_time = dtm.now()
time_str = current_time.strftime("%H:%M:%S")
with open('infere-log.txt', 'w') as f:
    f.write('The code started: ' + time_str + '\n')


sig1 = np.load('deconvolved-calcium-1.npy')
sig2 = np.load('deconvolved-calcium-2.npy')

signal = np.concatenate((sig1, sig2), axis=1)

del sig1, sig2

np.save('deconvolved-calcium.npy', signal)

#Connectivity Inference
####################################
G = np.load('connectivity-10e7-ms.npy')
G = G - (np.diag(np.diag(G)))

k = 10
Y = signal[:, k:]
Y_prime = signal[:, :-k]

yk = Y.T
y_k = Y_prime.T

reg = LinearRegression(n_jobs=-1).fit(y_k, yk)
a = reg.coef_
a = a - (np.diag(np.diag(a)))

corr_result = np.corrcoef(G.flatten(), a.flatten())[0, 1]

current_time = dtm.now()
time_str = current_time.strftime("%H:%M:%S")
with open('infere-log.txt', 'a') as f:
    f.write('\n' + 'Correlation coefficient between ground trouth and estimated connectivity is: ' + str(corr_result))

