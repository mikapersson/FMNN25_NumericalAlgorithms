from numpy import *
from OptimizationProblem import *
import matplotlib.pyplot as plt
from Newton import *
from QuasiNewton import *
from linesearchmethods import inexact_linesearch
from chebyquad_problem import *
import scipy.optimize as opt


def rosenbrock(x):  # optimal solution is (1,1)
    if len(x) != 2:
        raise ValueError("Rosenbrock takes arguments from R^2, not R^{}".format(len(x)))
        return

    x_1 = x[0]
    x_2 = x[1]
    return 100*(x_2 - x_1**2)**2 + (1 - x_1)**2


def contour_rosenbrock(levels=100, optipoints=array([])):
    """
    Plots the contours of the rosenbrock function, alternatively together with the optimization points
    :param levels: (int) number of level curves we want to display
    :param optipoints: (array)
    :return: None
    """

    # Verifying that 'optipoints' has the correct shape
    if optipoints.shape == (0,):
        pass
    elif optipoints.ndim == 2 and len(optipoints) == 2:  # optipoints is an ndarray with 2 rows
        pass
    else:
        raise ValueError("\'optipoints\' must have exactly 2 rows")

    size = 1000
    x = linspace(-0.5, 2, size)
    y = linspace(-1.5, 4, size)
    X, Y = meshgrid(x, y)
    input = array([X, Y])
    Z = rosenbrock(input)

    if len(optipoints) != 0:  # plot optimization points (either optipoints is empty or has 2 columns)
        plt.scatter(optipoints[0, :], optipoints[1, :])

    plt.contour(X, Y, Z, levels)
    plt.title("Contour plot of Rosenbrock's function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def simple_function(x):
    value = 3*sin(x[0]) + sin(x[1])
    return value


def gradient(x):
    grad = cos(x[0])+cos(x[1])
    return grad


def task7():
    x = array([[0],
               [3]])
    rho = 0.1
    sigma = 0.7
    tau = 0.1
    chi = 9
    problem = OptimizationProblem(rosenbrock)
    method = QuasiNewton(problem)
    s = method.gradient(x)
    res = inexact_linesearch(rosenbrock, x, s, rho, sigma, tau, chi)
    print(res)


def newton_methods(problem=None):
    """
    Test Quasi Newton methods on the (optionally specified) 'problem'
    :param problem:
    :return:
    """

    # Let the user choose method
    method = input("Testing Newton methods, choose one of the following:\n\t\'newton\', \'goodBroyden\', \'badBroyden\', \'symmetricBroyden\', \'DFP\', \'BFGS\'\nMethod: ")

    # Check that the chosen method is valid
    valid_methods = {"newton": Newton, "goodBroyden": GoodBroyden, "badBroyden": BadBroyden
                        , "symmetricBroyden": SymmetricBroyden, "DFP": DFP, "BFGS": BFGS}
    while method not in valid_methods:
        print("\nMethod \'{}\' does not exist, choose one of the following:\n\t\'newton\', \'goodBroyden\', \'badBroyden\', \'symmetricBroyden\', \'DFP\', \'BFGS\'".format(method))
        method = input("Method: ")

    if not problem:  # if we didn't specify a problem we take the 'simple_function'
        print("Using default problem")
        problem = OptimizationProblem(simple_function)

    # Choose linesearch method
    lsm = input("\nChoose line search method:\t\'exact\' or \'inexact\'\nLSM: ")

    # Create solver-instance
    try:
        solver = valid_methods[method](problem, lsm)
    except ValueError:
        lsm = input("\nLSM \'{}\' does not exist, choose between \'exact\' and \'inexact\'\nLSM: ".format(lsm))
        solver = valid_methods[method](problem, lsm)

    print("\nRunning {} method with {} line search".format(method, lsm))
    min_point, min_value = solver.solve()
    optipoints = solver.values
    print("\nOptimal point:\n", min_point)
    print("\nMinimum value:\n", min_value)
    # contour_rosenbrock(optipoints=optipoints)  # uncomment to plot rosenbrock_contour and optimization points


def chebyquad():
    problem = OptimizationProblem(chebyquad)

    scalar = 1
    ns = [4, 8, 11]
    for n in ns:
        solution = Newton(problem)
        min_point, min_value = solution.solve()
        min2 = opt.fmin(chebyquad, ones((n, 1)) * scalar, gradchebyquad)


def main():
    newton_methods()


main()