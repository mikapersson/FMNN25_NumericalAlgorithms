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

        for i in range(0, size):  # Calculate the spline self.su
            self.su[i] = self.__call__(i * len(self.knot_points) / size)

    def __call__(self, u):  # follow 1.9 summary
        # Find hot interval. Index of the element with higher value (than u) - 1.
        I = (self.knot_points > u).argmax() - 1

        if I == 0:
            return self.control_points[0]
        if I == -1:
            return self.control_points[-1]

        # Select 4 control points d_{I-2} to d_{I+1}
        four_control = self.control_points[I-2:I+2].copy()

        # Run blossom algorithm (de Boor)
        s = self.Blossoms(four_control, self.knot_points, I, u)  # d[u,u,u] = s(u)
        return s

    def Blossoms(self, c, k, I, u):
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
