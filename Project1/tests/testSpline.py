from basis_function import basis_function
from CubicSpline import *
import unittest

"""
Authors: Mika Persson & Viktor Sambergs
"""


class TestSpline(unittest.TestCase):
    test_control = array(
        [[-17, 9],
         [-40, 20],
         [-20, 0],
         [-100, -15],
         [-22, -62],
         [8, -78],
         [57, -30],
         [15, 8],
         [18, -3],
         [40, 17]])
    test_knots = array([0, 0, 0, 1, 2, 3, 6, 9, 11, 12, 12, 12])
    inp1 = (test_control, test_knots)

    def testinit(self):
        spline = CubicSpline(*self.inp1)
        assert (self.test_knots == spline.knot_points).all()
        assert (self.test_control == spline.control_points).all()

    def test_unity(self):
        """
        Tests if the basis functions sum up to one
        """

    def test_bspline(control, knots):
        control = array(control)
        size = 1000
        s = zeros((size, 2))
        basis_functions = array([basis_function(size, knots, i) for i in range(len(knots) - 2)])
        for u in range(size):
            # Find hot interval. Index of the element with higher value (than u) - 1.
            I = (knots > u).argmax() - 1

            if I == 0:
                s[u] = knots[0]
            if I == -1:
                s[u] = knots[-1]

            s[u] = sum(
                control[i] * basis_functions[i, u] for i in range(I - 2, I + 2))  # change u in basis_function-argument

        plt.plot(s[:, 0], s[:, 1], 'b', label="cubic spline")  # spline
        plt.plot(control[:, 0], control[:, 1], '-.r', label="control polygon")  # control polygon
        plt.scatter(control[:, 0], control[:, 1], color='red')  # de Boor points

        plt.title("Cubic Spline")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid()
        plt.show()


if __name__=='__main__':
    unittest.main()