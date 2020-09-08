import matplotlib.pyplot as plt
from numpy import *
import scipy.linalg as sl
from greville_abscissae import greville_abscissae
from vandermonde import vandermonde

"""
Authors: 
"""


class CubicSpline:
    """
    Class for creating and plotting a cubic spline
    """

    def __init__(self, control, knots):
        """
        Constructs a cubic spline according to the given control points and knots
        :param control: (array)
        :param knots: (array)
        """

        self.control_points = control
        # Add d_{-2}, d_{-1}, d_{K+1} and d_{K+2}
        self.control_points = vstack([control[0], control[0], control, control[-1], control[-1]])

        self.knot_points = knots

        size = 10000
        self.su = zeros((size, 2))

        for i in range(0, size):  # Calculate the spline self.su
            self.su[i] = self.__call__(i * len(self.knot_points) / size)

    def __call__(self, u):  # follows 1.9 summary
        """
        Computes and returns the value of the spline at 'u'
        :param u: (float)
        :return: (float)
        """

        # Find hot interval. Index of the element with higher value (than u) - 1.
        I = (self.knot_points > u).argmax() - 1

        if I == 0:
            return self.control_points[0]
        if I == -1:
            return self.control_points[-1]

        # Select 4 control points d_{I-2} to d_{I+1}
        four_control = self.control_points[I-2:I+2].copy()

        # Run blossom algorithm (de Boor)
        s = self.blossoms(four_control, self.knot_points, I, u)  # d[u,u,u] = s(u)
        return s

    def interpolate(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """

        vander = vandermonde(self.knot_points)
        dx = sl.solve(vander, x)
        dy = sl.solve(vander, y)

        return dx, dy

    def blossoms(self, c, k, I, u):
        """
        Evaluates s(u) in accordance with De Boor's algorithm (section 1.7 in the slides)
        :param c: (array) Array of "hot" control points of shape (4,)
        :param k: (array) Array of knot points of shape (#knot points + 4,)
        :param I: (int) Index of interval that contains u
        :param u: (int) position in [u_0, u_K]
        :return: (array) the value of the spline at the knot point u, the shape is (2,)
        """

        # Size of array of "hot control points" (number of elements in a column in slide 8)
        size = c.shape[0]

        # If the size is 1 c is s(u) (d[u,u,u])
        if size == 1:
            return c

        # Array for storing new control points
        # Should be size (3,2), then (2,2), lastly (1,2)
        newcontrolpoints = zeros((size - 1, 2))

        for i in range(1, size):
            # At first call operations are performed 3 times, then 2, then one.

            # New control point at index i is constructed from previous points at index i and i+1

            # Old control points have knot points spanning over a range size-1 (4 indices, 3 indices and so on).
            r = size - 1

            # Max index for knotpoints is I+i, min is (I+i)-range.
            maxindex = I + i
            minindex = maxindex - r

            # Leftmost- (u_{(I+i)-r}) and rightmost knot (u_{I+i}) for construction of alpha(u)
            urightmostknot = k[maxindex]
            uleftmostknot = k[minindex]

            alphau = (urightmostknot - u) / (urightmostknot - uleftmostknot)

            # Vector of new control points is filled.
            newelement = alphau * c[i - 1] + (1 - alphau) * c[i]

            newcontrolpoints[i - 1] = newelement

        # For every call the size of newcontrolpoints is smaller until s(u) is returned.
        return self.blossoms(newcontrolpoints, k, I, u)

    def plot(self):
        """
        Plot the cubic spline (self.s_u), its control polygon (generated by (self.control_points)
        and the de Boor points (self.control_points)
        :return: None
        """

        plt.plot(self.su[:, 0], self.su[:, 1], 'b', label="cubic spline")  # spline
        plt.plot(self.control_points[:, 0], self.control_points[:, 1], '-.r', label="control polygon")  # control polygon
        plt.scatter(self.control_points[:, 0], self.control_points[:, 1], color='red')  # de Boor points

        plt.title("Cubic Spline")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid()
        plt.show()