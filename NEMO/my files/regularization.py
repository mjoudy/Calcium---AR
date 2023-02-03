import numpy as np
from sklearn.linear_model import Lasso

spikes = np.load("spikes-10e5-ms-n100.npy")

k = 10
Y = spikes[:, k:]
Y_prime = spikes[:, :-k]

G = np.load('connectivity-10e5-ms-n100.npy')
G = G - (np.diag(np.diag(G)))

yk = Y.T
y_k = Y_prime.T

lamda = 0.00375
las = Lasso(alpha = lamda)
las_reg = las.fit(y_k, yk)

b = las.coef_
b = b - (np.diag(np.diag(b)))
np.corrcoef(G.flatten(), b.flatten())[0, 1]
