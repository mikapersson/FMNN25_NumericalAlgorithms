from numpy import *


def basis_function(knot_points, j):
    """
    Returns the j:th B-spline basis function (N_j)^3 to the corresponding 'knot_points'
    :param knot_points: (array)
    :param j: (int) index of basis function to be returned
    :return: (array) j:th basis function of 'knot_points'
    """

    basis_knots = insert(knot_points, [0, len(knot_points)], [knot_points[0], knot_points[-1]])  # insert u_{-1} and u_{K+1}

