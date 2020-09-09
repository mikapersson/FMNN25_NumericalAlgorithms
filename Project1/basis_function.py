from numpy import *

"""
@author: Mika Persson & Viktor Sambergs
"""


def basis_function(size, knots, j):
    """
    Returns the i:th B-spline basis function with 'size' values, given the 'knots'
    :param size: (int) number of values in the function
    :param knots: (array)
    :param j: (int) index of basis function
    :return: (array) the i:th B-spline basis function with 'size' values, given the 'knots'
    """

    result = zeros(size)
    for i in range(0, size):
        result[i] = basis_function_rec(knots, j, 3, i * knots[-1] / (size-1))

    return result


def basis_function_rec(knots, i, k, u):
    """
    Returns the i:th B-spline basis function of order 'k' given the 'knots' at 'u'
    :param knots: (array)
    :param i: (int) index of basis function
    :param k: (int) order of the basis function
    :param u: (float) at which point we evaluate the value
    :return: (array) the i:th B-spline basis function of order 'k' given the 'knots' at 'u'
    """

    if k == 3 and i == len(knots) - 3 and knots[-1] == u:
        return 1

    extended_knots = knots
    if k == 3:
        extended_knots = insert(extended_knots, len(knots),  knots[-1])

    if k == 0:
        return 1.0 if extended_knots[i-1] <= u < extended_knots[i] else 0.0
    if extended_knots[i + k - 1] == extended_knots[i-1]:
        c1 = 0.0
    else:
        c1 = (u - extended_knots[i-1]) / (extended_knots[i-1 + k] - extended_knots[i-1]) * basis_function_rec(extended_knots, i, k-1, u)
    if extended_knots[i + k] == extended_knots[i]:
        c2 = 0.0
    else:
        c2 = (extended_knots[i + k] - u) / (extended_knots[i + k] - extended_knots[i]) * basis_function_rec(extended_knots, i+1, k-1, u)
    return c1 + c2
