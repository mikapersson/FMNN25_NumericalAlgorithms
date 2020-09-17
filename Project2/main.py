from numpy import *
from OptimizationProblem import*
import matplotlib.pyplot as plt
from QuasiNewton import*


def rosenbrock(x):
    if len(x) != 2:
        raise ValueError("Rosenbrock takes arguments from R^2, not R^{}".format(len(x)))
        return

    x_1 = x[0]
    x_2 = x[1]
    return 100*(x_2 - x_1**2)**2 + (1 - x_1)**2


def contour_rosenbrock(levels=100, optipoints=array([])):
    if optipoints.shape == (0,) or optipoints.ndim == 2:
        pass
    else:
        raise ValueError("\'optipoints\' must have exactly 2 columns")

    size = 1000
    x = linspace(-0.5, 2, size)
    y = linspace(-1.5, 4, size)
    X, Y = meshgrid(x, y)
    input = array([X, Y])
    Z = rosenbrock(input)

    if len(optipoints) != 0:  # plot optimization points (either optipoints is empty or has 2 columns)
        plt.scatter(optipoints[:, 0], optipoints[:, 1])

    plt.contour(X, Y, Z, levels)
    plt.title("Contour plot of Rosenbrock's function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def function(x):
    value = 3*sin(x[0]) + sin(x[1])
    return value


def gradient(x):
    grad = cos(x[0])+cos(x[1])
    return grad


# problem = OptimizationProblem(function, gradient)
#solution = QuasiNewton(problem)
#a = solution.solve()
#print(a)

contour_rosenbrock()