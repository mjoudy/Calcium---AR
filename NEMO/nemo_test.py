import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

spikes = np.load("spikes-10e5-ms-n100.npy")

k = 10
Y = spikes[:, k:]
Y_prime = spikes[:, :-k]
Y_pinv = np.linalg.pinv(Y_prime)
#del Y_prime

A = Y @ Y_pinv
#del Y_pinv, Y

G = np.load('connectivity-10e5-ms-n100.npy')
G = G - (np.diag(np.diag(G)))

A = A - (np.diag(np.diag(A)))

corr = np.corrcoef(G.flatten(), A.flatten())[0, 1]

np.save('corr', corr)