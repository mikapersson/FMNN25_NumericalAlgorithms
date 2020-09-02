from scipy.linalg import *
from numpy import *
import matplotlib.pyplot as plt
from greville_abscissae import greville_abscissae

x = linspace(0, 10, 100)
a = random.rand(len(x))
y = greville_abscissae(a)

plt.plot(x, a)
plt.plot(x, y)
plt.grid()
plt.show()