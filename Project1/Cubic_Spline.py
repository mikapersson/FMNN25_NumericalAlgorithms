import matplotlib.pyplot as plt
from numpy import *
import scipy.linalg as sl
from greville_abscissae import greville_abscissae


class Cubic_Spline:
    """ SOMETHING ABOUT THE CLASS"""

    def __init__(self, control, knots):
        self.control_points = control
        self.knot_points = knots
        self.s_u = array([[self.control_points[0]]])

        # Construct self.s_u
        u_interval = linspace(self.knot_points[0], self.knot_points[-1], 100)
        for u in u_interval:
            vstack([self.s_u, self.__call__(u)])

    def __call__(self, u):  # follow 1.9 summary

        # 1) Find hot interval, index of least upper bound of u - 1
        I = (self.knot_points > u).argmax() - 1
        print("We found knot point {} at index {}".format(self.knot_points[I], I))

        # 2) Select 4 control points
        four_control = self.control_points[I-2:I+2].copy()

        # 3) Run blossom algorithm (de Boor)
        for r in range(1, 4):
            for j in range(3, r-1, -1):
                alpha = (four_control[-1] - u)/(four_control[-1] - four_control[0])

        # return d[u,u,u]

    def plot(self):
        """
        Plot the cubic spline (self.s_u), its control polygon and the de Boor points
        :return: None
        """
        plt.plot(self.s_u[:, 0], self.s_u[:, 1], 'b')
        plt.plot(self.control_points[:, 0], self.control_points[:, 1], '-.r')  # control polygon
        plt.scatter(self.control_points[:, 0], self.control_points[:, 1], color='red')  # de Boor points

        plt.grid()
        plt.show()