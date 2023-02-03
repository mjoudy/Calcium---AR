import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

spikes = np.load("calcium-60e6-tau-10.npy")

k = 10
Y = spikes[:, k:]
Y_prime = spikes[:, :-k]

G = np.load('connectivity-10e5-ms.npy')
G = G - (np.diag(np.diag(G)))

yk = Y.T
y_k = Y_prime.T

print("started at: ", datetime.now())

reg = LinearRegression(n_jobs=30).fit(y_k, yk)

print("ended at: ", datetime.now())

a = reg.coef_

a = a - (np.diag(np.diag(a)))

corr = np.corrcoef(G.flatten(), a.flatten())[0, 1]

print("correlation: ", corr)