from scipy.linalg import *
from numpy import *
import matplotlib.pyplot as plt
from greville_abscissae import greville_abscissae

a = array([1,2,3,4,5,6])
y = a[:-1:2j]
print(y)