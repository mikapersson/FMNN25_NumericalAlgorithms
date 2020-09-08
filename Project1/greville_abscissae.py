from numpy import *

"""
Assignment 1
authors: Mika Persson (mi3301pe-s@student.lu.se) and Viktor Sambergs (vi@student.lu.se)
"""


def greville_abscissae(knots):
    """
    Computes the moving average (order 3) of the knot points u
    :param u: (array) knot points
    :return: (array) moving average, same size as u
    """

    return (knots[:-2] + knots[1:-1] + knots[2:]) / 3
