from numpy import *

"""
Assignment 1
authors: Mika Persson (mi3301pe-s@student.lu.se) and Viktor Sambergs (vi@student.lu.se)
"""


def greville_abscissae(u):
    """
    Computes the moving average (order 3) of the knot points u
    :param u: (array) knot points
    :return: (array) moving average, same size as u
    """

    n = 3  # order of moving average
    ga = u.astype(float).copy()
    temp_array = cumsum(u, dtype=float)
    temp_array[n:] = temp_array[n:] - temp_array[:-n]
    ga[1:-1] = temp_array[n-1:]/n
    return ga
