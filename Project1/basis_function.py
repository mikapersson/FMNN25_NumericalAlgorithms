from numpy import *


def basis_function(knots, i, k, u):
    """
    Returns
    :param knots: (array)
    :param i: (int)
    :param k: (int) order of the basis function
    :param u: (float)
    :return: (array)
    """

    extended_knots = knots
    if k == 3:  # if we've just entered the recursion
        # Add u_{-2}, u_{-1}, u_{K+1} and u_{K+2}
        extended_knots = insert(extended_knots, [0, 0, len(knots), len(knots)], [knots[0], knots[0], knots[-1], knots[-1]])

    if k == 0:
        return 1.0 if extended_knots[i-1] <= u < extended_knots[i] else 0.0
    if extended_knots[i + k - 1] == extended_knots[i-1]:
        c1 = 0.0
    else:
        c1 = (u - extended_knots[i-1]) / (extended_knots[i-1 + k] - extended_knots[i-1]) * basis_function(extended_knots, i, k-1, u)
    if extended_knots[i + k] == extended_knots[i]:
        c2 = 0.0
    else:
        c2 = (extended_knots[i + k] - u) / (extended_knots[i + k] - extended_knots[i]) * basis_function(extended_knots, i+1, k-1, u)
    return c1 + c2
