import numpy as np
from sklearn.linear_model import Lasso
from datetime import datetime

spikes = np.load("calcium-10e4-tau-10.npy")

k = 10
Y = spikes[:, k:]
Y_prime = spikes[:, :-k]

G = np.load('connectivity-10e4-ms.npy')
G = G - (np.diag(np.diag(G)))

yk = Y.T
y_k = Y_prime.T

t1 = datetime.now()
print ("start: ", t1)

lamda = 0.00375
las = Lasso(alpha = lamda)
las_reg = las.fit(y_k, yk)

b = las.coef_
b = b - (np.diag(np.diag(b)))
cor = np.corrcoef(G.flatten(), b.flatten())[0, 1]

print ("lambda: ", lamda)
print ("correlation coef: ", cor)

t2 = datetime.now()
print ("end: ", t2)
