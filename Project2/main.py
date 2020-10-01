from OptimizationProblem import *
import matplotlib.pyplot as plt
from QuasiNewton import *
from linesearchmethods import inexact_linesearch
from chebyquad_problem import *
import scipy.optimize as opt

"""
Main file for manual testing of the quasi Newton solvers
"""


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
    x = linspace(-0.5, 5, size)
    y = linspace(-1.5, 16, size)
    X, Y = meshgrid(x, y)
    input = array([X, Y])
    Z = rosenbrock(input)

    if len(optipoints) != 0:  # plot optimization points (either optipoints is empty or has 2 rows)
        plt.scatter(optipoints[0, :], optipoints[1, :])

    plt.contour(X, Y, Z, levels)
    plt.title("Contour plot of Rosenbrock's function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


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


def newton_methods_test(problem=None):
    """
    Test Quasi Newton methods on the (optionally specified) 'problem'
    """

    # Let the user choose function if the problem wasn't already specified
    if not problem:
        function_name = input("Choose function to optimize:\n\t\'sinus\', \'paraboloid\' or \'rosenbrock\' (dangerous)"
                         "\nFunction: ")
        valid_functions = {"sinus": sin_function, "paraboloid": paraboloid_function, "rosenbrock": rosenbrock}
        while function_name not in valid_functions:
            print("\nFunction \'{}\' does not exist, choose one of the following:\n\'sinus\', \'paraboloid\' or "
                  "\'rosenbrock\'".format(function_name))
            function_name = input("Function: ")

        function = valid_functions[function_name]  # converts from string to actual function

        print("Using default problem (default gradient and points in R^2)")
        problem = OptimizationProblem(function)

    # Let the user choose method
    method = input("\nTesting Newton methods, choose one of the following:\n\t\'newton\', \'goodBroyden\', "
                   "\'badBroyden\', \'symmetricBroyden\', \'DFP\', \'BFGS\'\nMethod: ")

    # Check that the chosen method is valid
    valid_methods = {"newton": Newton, "goodBroyden": GoodBroyden, "badBroyden": BadBroyden
                        , "symmetricBroyden": SymmetricBroyden, "DFP": DFP, "BFGS": BFGS}
    while method not in valid_methods:
        print("\nMethod \'{}\' does not exist, choose one of the following:\n\t\'newton\', \'goodBroyden\', "
              "\'badBroyden\', \'symmetricBroyden\', \'DFP\', \'BFGS\'".format(method))
        method = input("Method: ")

    # Choose line search method
    lsm = input("\nChoose line search method:\t\'exact\' or \'inexact\'\nLSM: ")

    # Create solver-instance
    try:
        solver = valid_methods[method](problem, lsm)
    except ValueError:
        lsm = input("\nLSM \'{}\' does not exist, choose between \'exact\' and \'inexact\'\nLSM: ".format(lsm))
        solver = valid_methods[method](problem, lsm)

    # Solve problem
    print("\nOptimizing {} with {} method with {} line search".format(function_name, method, lsm))
    min_point, min_value = solver.solve()
    print("\nOptimal point:\n", min_point)
    print("\nMinimum value:\n", min_value)

    if function_name == "rosenbrock":
        optipoints = solver.values
        contour_rosenbrock(optipoints=optipoints)


def chebyquad_test():
    """
    Testing optimization of the Chebyquad-problem of a chosen degree n and comparing the results with those from numpy.optimize.fmin_bfgs().
    """

    degree = input("Testing mimimization of the Chebyquad function of degree n. Choose degree n: ")
    degree = int(degree)

    method = input(
        "Testing Newton methods, choose one of the following:\n\t\'newton\', \'goodBroyden\', \'badBroyden\', \'symmetricBroyden\', \'DFP\', \'BFGS\'\nMethod: ")

    # Check that the chosen method is valid
    valid_methods = {"newton": Newton, "goodBroyden": GoodBroyden, "badBroyden": BadBroyden
        , "symmetricBroyden": SymmetricBroyden, "DFP": DFP, "BFGS": BFGS}
    while method not in valid_methods:
        print(
            "\nMethod \'{}\' does not exist, choose one of the following:\n\t\'newton\', \'goodBroyden\', \'badBroyden\', \'symmetricBroyden\', \'DFP\', \'BFGS\'".format(
                method))
        method = input("Method: ")

    problem = OptimizationProblem(chebyquad, dimension=degree)
    solver = valid_methods[method](problem, lsm="inexact")
    a = solver.solve()

    print()
    print("Optimizing the Chebyquad function of degree n = " + str(degree) + " with a " + str(method) + "-method: ")
    print("Function value: " + str(a[1]))
    print("The minimum was found in: " + str(transpose(a[0])))
    print()
    print("Optimizing the same function with numpy.optimize.fmin: ")
    min2 = opt.fmin_bfgs(chebyquad, ones((degree, 1)) * 1, disp=False, full_output=True)
    print("Function value: " + str(min2[1]))
    print("The minimum was found in: " + str(min2[0]))
    print()

    if isclose(min2[0], transpose(a[0])).all() and isclose(min2[1], a[1]):
        print("The two methods yield the same result.")
    else:
        print("The two methods do NOT yield the same result.")


def HessianQualityControl():
    """
    Testing quality of the evaluation of the inverse Hessian by comparing mean differences between calculated and exact
    matrices.
    """

    problem = OptimizationProblem()
    solver = BFGS(problem, lsm="inexact", hessians="on")
    solution = solver.solve()

    def true_hessian_inverse(x):
        """The exact inverse hessian"""
        hessian = [1/(-3*sin(x[0])), 0, 0, 1/-sin(x[1])]
        hessian = reshape(hessian,[-1])
        return hessian

    nmbr = size(solution[1])  # number of evaluations k
    inverse_hessians = reshape(solution[0],[-1])  # Elements of all calculated hessians
    allhess = empty(0)
    differences = empty(0)
    k = 1

    for i in range(0,nmbr,2):
        coordinates = solution[1][i:i+2]  # Coordinates of calculated hessians
        print(coordinates)
        truehess = true_hessian_inverse(coordinates)  # Calculating exact hessian
        allhess = append(allhess,truehess)

    while inverse_hessians.size > 1:
        # Comparing values of hessian elements
        testvalues = inverse_hessians[0:4]
        truevalues = allhess[0:4]

        difference = abs(testvalues-truevalues)
        # print("k = " + str(k))
        k +=1
        differences = append(differences, mean(difference))

        inverse_hessians = inverse_hessians[4:-1]
        allhess = allhess[4:-1]

    x = linspace(1,k-1,k-1)
    plt.plot(x,differences)
    plt.xlabel("k")
    plt.ylabel("Mean difference between calculated and exact inverse Hessian")
    plt.show()



def main():
    newton_methods_test()
    # chebyquad_test()
    # HessianQualityControl()


main()