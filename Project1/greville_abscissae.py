from numpy import *

"""
Assignment 1
@author: Mika Persson & Viktor Sambergs
"""


def greville_abscissae(knots):
    """
    Computes the moving average (order 3) of the knot points u
    :param u: (array) knot points
    :return: (array) moving average, same size as u
    """

    return (knots[:-2] + knots[1:-1] + knots[2:]) / 3
