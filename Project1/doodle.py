import matplotlib.pyplot as plt
from numpy import *
import scipy.linalg as sl
from greville_abscissae import greville_abscissae

test_control = array([[-40, 20], [-20, 0], [-100, -15], [-22, -62], [8, -78], [57, -30], [15, 8], [18, -3], [40, 17]])
plt.plot(test_control[:,0], test_control[:,1], '-.b')
plt.scatter(test_control[:,0], test_control[:,1])
plt.show()
