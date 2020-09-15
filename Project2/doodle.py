import matplotlib.pyplot as plt
from numpy import *
import scipy.linalg as sl


def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


h = 0.0001
x = array([1, 1.1])
n = len(x)


def g(x):  # Calculates the gradient of the objective function f, for given
    g = zeros(n)  # values of x1,..,xn. The vector e represents the indexes where the
    for i in range(n):  # stepsize, h should be added depending on what partial derivative we want
        e = zeros(n)  # to caclulate
        e[i] = h
        g[i] = (rosenbrock(x + e) - rosenbrock(x)) / h
    return g

res = g(x)