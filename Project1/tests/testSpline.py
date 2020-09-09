from basis_function import basis_function, basis_function_rec
from CubicSpline import *
import matplotlib.pyplot as plt
import unittest
from numpy.testing import assert_allclose

"""
Test of Spline Class

@author: Mika Persson & Viktor Sambergs
"""


class TestSpline(unittest.TestCase):
    """
    Test CubicSpline object and methods
    """
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
    x = array(test_control[:, 0])
    y = array(test_control[:, 1])

    def testinit(self):
        """
        Test Cubic Spline init
        """
        spline = CubicSpline(*self.inp1)
        assert (self.test_knots == spline.knot_points).all()
        assert (self.test_control == spline.control_points).all()

    def testunity(self):
        """
        Tests unity of basis functions
        """
        grid = self.test_knots
        elements = zeros(size(grid))
        for u in range(grid[0], grid[-1]):
            for i in range(size(grid)-3):
                elements[i] = basis_function_rec(grid, i, 3, u)
            s = sum(elements)
            unity = 1
            self.assertAlmostEqual(s, unity)

    def test_bspline(self):  # THIS TEST DOESN'T QUITE PASS
        """
        Test if Cubic Spline created from object CubicSpline equals the sum of the control points
        and basis functions as in section 1.5
        """
        cubsplin = CubicSpline(self.test_control, self.test_knots)

        size = 10000
        bspline = zeros((size, 2))
        basis_functions = array([basis_function(size, self.test_knots, j) for j in range(len(self.test_knots) - 2)])
        for u in range(size):
            bspline[u] = sum(self.test_control[i] * basis_functions[i, u] for i in range(len(basis_functions)))

        assert_allclose(cubsplin.get_spline(), bspline)  # ERROR

    def test_interpolation(self):
        """
        Test if the first/last spline point equals the first/last interpolation point
        """
        cubsplin = CubicSpline(*self.inp1)
        first_inter_point = array([self.x[0], self.y[0]])
        last_inter_point = array([self.x[-1], self.y[-1]])
        first_spline_point = cubsplin.get_spline()[0]
        last_splint_point = cubsplin.get_spline()[-1]

        assert (first_inter_point == first_spline_point).all()
        assert (last_inter_point == last_splint_point).all()


if __name__=='__main__':
    unittest.main()