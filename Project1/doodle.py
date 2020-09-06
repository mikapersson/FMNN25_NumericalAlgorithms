import matplotlib.pyplot as plt
from numpy import *
import scipy.linalg as sl
from greville_abscissae import greville_abscissae

a = array([[0, 1], [1, 2], [2, 3]])
a = vstack([a[0], a])
print(a)
