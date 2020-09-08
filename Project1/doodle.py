import matplotlib.pyplot as plt
from numpy import *
import scipy.linalg as sl
from greville_abscissae import greville_abscissae

u = arange(10, dtype=float)
print(greville_abscissae(u))