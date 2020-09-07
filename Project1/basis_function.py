from numpy import *


def basis_function(size, knots, j):
    """
    Returns the i:th B-spline basis function with 'size' values, given the 'knots'
    :param size: (int) number of values in the function
    :param knots: (array)
    :param j: (int) index of basis function
    :return: (array) the i:th B-spline basis function with 'size' values, given the 'knots'
    """

    extended_knots = knots
    # Add u_{-2}, u_{-1}, u_{K+1} and u_{K+2}
    extended_knots = insert(extended_knots, [0, 0, len(knots), len(knots)], [knots[0], knots[0], knots[-1], knots[-1]])

    result = zeros(size)
    for i in range(0, size):
        result[i] = basis_function_rec(extended_knots, j, 3, i * extended_knots[-1] / size)

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

    if k == 0:
        return 1.0 if knots[i-1] <= u < knots[i] else 0.0
    if knots[i + k - 1] == knots[i-1]:
        c1 = 0.0
    else:
        c1 = (u - knots[i-1]) / (knots[i-1 + k] - knots[i-1]) * basis_function_rec(knots, i, k-1, u)
    if knots[i + k] == knots[i]:
        c2 = 0.0
    else:
        c2 = (knots[i + k] - u) / (knots[i + k] - knots[i]) * basis_function_rec(knots, i+1, k-1, u)
    return c1 + c2
