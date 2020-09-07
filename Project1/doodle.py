import matplotlib.pyplot as plt
from numpy import *
import scipy.linalg as sl
from greville_abscissae import greville_abscissae

size = 100
k = arange(1, size)
a = [(-1-1/k)**k for k in range(1, size)]

plt.plot(k, a)
plt.grid()
plt.show()
