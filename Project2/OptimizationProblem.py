# Its constructor should take an
# objective function as input, optionally also its gradient can be provided.

from numpy import sin, cos


# We define some functions that we later use for testing
def sin_function(x):
    if len(x) != 2:
        raise ValueError("Sinus function takes arguments from R^2, not R^{}".format(len(x)))

    value = 3*sin(x[0]) + sin(x[1])
    return value


def paraboloid_function(x):
    if len(x) != 2:
        raise ValueError("Paraboloid function takes arguments from R^2, not R^{}".format(len(x)))

    return x[0]**2 + x[1]**2


def gradient(x):
    grad = cos(x[0])+cos(x[1])
    return grad


def rosenbrock(x):  # optimal solution is (1,1)
    if len(x) != 2:
        raise ValueError("Rosenbrock takes arguments from R^2, not R^{}".format(len(x)))

    x_1 = x[0]
    x_2 = x[1]
    return 100*(x_2 - x_1**2)**2 + (1 - x_1)**2


class OptimizationProblem:
    """
    Defines a problem, independent of solution method.
    """

    def __init__(self, function=sin_function, gradient=None, dimension=2):
        self.function = function
        self.gradient = gradient
        self.dimension = dimension
