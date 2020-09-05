import matplotlib.pyplot as plt
from numpy import *
import scipy.linalg as sl
from greville_abscissae import greville_abscissae


class Cubic_Spline:
    """ SOMETHING ABOUT THE CLASS"""

    def __init__(self, knots):
        self.knot_points = knots
        self.control_points = array([])
        self.s_u = array([])

    def __call__(self, u):  # follow 1.9 summary

        # 1) Find hot interval
        I = self.knot_points.searchsorted([u]) - 1

        # 2) Select 4 control points

        # 3) run blossom algorithm

        # return s_u
        grev_abcsi = greville_abscissae(self.knot_points)

    def plot(self):
        x = linspace(0, 1, len(self.knot_points))
        plt.plot(x, self.knot_points)
        plt.grid()
        plt.show()