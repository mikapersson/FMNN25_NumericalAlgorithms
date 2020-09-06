import matplotlib.pyplot as plt
from numpy import *
import scipy.linalg as sl
from greville_abscissae import greville_abscissae

a = array([1,2,3,4])
a = insert(a, [0, len(a)], [0, 5])
print(a)
