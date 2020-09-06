import matplotlib.pyplot as plt
from numpy import *
import scipy.linalg as sl
from greville_abscissae import greville_abscissae


class Cubic_Spline:
    """ SOMETHING ABOUT THE CLASS"""

    def __init__(self, control, knots):
        self.control_points = control
        # Add d_{-2}, d_{-1}, d_{K+1} and d_{K+2}
        self.control_points = vstack([control[0], control[0], control, control[-1], control[-1]])

        self.knot_points = knots
        # Add u_{-2}, u_{-1}, u_{K+1} and u_{K+2}
        self.knot_points = insert(self.knot_points, [0, 0, len(self.knot_points), len(self.knot_points)], [knots[0], knots[0], knots[-1], knots[-1]])

        size = 500
        self.su = zeros((size, 2))

        for i in range(0, size):
            self.su[i] = self.__call__(i * len(self.knot_points) / size)

        '''
        self.s_u = array([[self.control_points[0]]])

        # Construct self.s_u
        u_interval = linspace(self.knot_points[0], self.knot_points[-1], 100)
        for u in u_interval:
            vstack([self.s_u, self.__call__(u)])
        '''

    def __call__(self, u):  # follow 1.9 summary
        # Find hot interval. Index of the element with higher value - 1.
        I = (self.knot_points > u).argmax() - 1

        if I == 0:
            return self.control_points[0]
        if I == -1:
            return self.control_points[-1]

        # 2) Select 4 control points d_{I-2} to d_{I+1}
        four_control = self.control_points[I-2:I+2].copy()

        # 3) Run blossom algorithm (de Boor)
        s = self.Blossoms(four_control, self.knot_points, I, u)  # d[u,u,u] = s(u)
        return s

    def Blossoms(self, c, k, I, u):
        """
        Evaluates s(u) in accordance with De Boor's algorithm
        :param c:
        :param k:
        :param I:
        :param u:
        :return:
        """

        # Size of array of "hot control points"
        size = c.shape[0]
        # Array for storing new control points

        # Should be size (1,2), then 2,2... (i,2)
        newcontrolpoints = zeros((size - 1, 2))

        # If the size is 1 c is S(u)

        if size == 1:
            return c

        for i in range(1, size):
            # At first call operations are performed 3 times, then 2, then one.

            # New control point at index i is constructed from previous points at index i and i+1

            # Old control points have knot points spanning over a range size-1 (4 indices, 3 indices and so on).
            r = size - 1

            # Max index for knotpoints is I+i, min is I+i-range.
            maxindex = I + i
            minindex = maxindex - r

            # Leftmost and rightmost knot for construction of alpha(u)
            urightmostknot = k[maxindex]
            uleftmostknot = k[minindex]

            alphau = (urightmostknot - u) / (urightmostknot - uleftmostknot)

            # Vector of new control points is filled.
            newelement = alphau * c[i - 1] + (1 - alphau) * c[i]

            newcontrolpoints[i - 1] = newelement

        # For every call the size of newcontrolpoints is smaller until s(u) is returned.
        return self.Blossoms(newcontrolpoints, k, I, u)

    def plot(self):
        """
        Plot the cubic spline (self.s_u), its control polygon and the de Boor points
        :return: None
        """
        plt.plot(self.su[:, 0], self.su[:, 1], 'b')  # spline
        plt.plot(self.control_points[:, 0], self.control_points[:, 1], '-.r')  # control polygon
        plt.scatter(self.control_points[:, 0], self.control_points[:, 1], color='red')  # de Boor points

        plt.grid()
        plt.show()
