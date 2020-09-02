from numpy import *

"""
Assignment 1
authors: Mika Persson (mi3301pe-s@student.lu.se) and Viktor Sambergs (vi@student.lu.se)
"""


def greville_abscissae(a):
    """

    :param a:
    :return:
    """

    n = 3
    ga = a.astype(float).copy()
    temp_array = cumsum(a, dtype=float)
    temp_array[n:] = temp_array[n:] - temp_array[:-n]
    ga[1:-1] = temp_array[n-1:]/n
    return ga