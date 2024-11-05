import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from datetime import datetime as dtm

current_time = dtm.now()
time_str = current_time.strftime("%H:%M:%S")
with open('lasso-log.txt', 'w') as f:
    f.write('The code started: ' + time_str + '\n')

signal = np.load("spikes-10e5-ms.npy")

current_time = dtm.now()
time_str = current_time.strftime("%H:%M:%S")
with open('lasso-log.txt', 'a') as f:
    f.write('spikes loaded: ' + time_str + '\n')

k = 10
Y = signal[:, k:]
Y_prime = signal[:, :-k]

yk = Y.T
y_k = Y_prime.T
las = Lasso(alpha=0.00375).fit(y_k, yk)
A = las.coef_

G = np.load('connectivity-10e5-ms.npy')
G = G - (np.diag(np.diag(G)))

A = A - (np.diag(np.diag(A)))

corr_result = np.corrcoef(G.flatten(), A.flatten())[0, 1]

current_time = dtm.now()
time_str = current_time.strftime("%H:%M:%S")
with open('lasso-log.txt', 'a') as f:
    f.write('lasso calculated: ' + time_str + '\n')
    f.write('\n' + 'Correlation coefficient between ground trouth and estimated connectivity is: ' + str(corr_result))
