from OptimizationProblem import *
from QuasiNewton import *
from linesearchmethods import *
import unittest
from numpy.testing import assert_almost_equal

"""
Test functions/methods used in Newton.py, QuasiNewton.py and linearsearchmethods.py

@author: Mika Persson & Viktor Sambergs
"""


class TestOptimization(unittest.TestCase):
    """
    The Rosenbrock function is used to test all solvers
    """

    difficult_problem = OptimizationProblem(rosenbrock)
    difficult_solution = array([[1],
                      [1]])

    easy_problem = OptimizationProblem(paraboloid_function)
    easy_solution = array([[0],
                           [0]])

    def test_gradient(self):
        pass

    def test_termination_criterion(self):
        pass

    def test_newton(self):
        newton_solver = Newton(self.difficult_problem)
        optimum, optimum_value = newton_solver.solve()
        assert_almost_equal(self.difficult_solution, optimum, decimal=5)

    def test_good_broyden(self):
        gb_solver = GoodBroyden(self.easy_problem)
        optimum, optimum_value = gb_solver.solve()
        assert_almost_equal(self.easy_solution, optimum, decimal=5)

    def test_bad_broyden(self):
        gb_solver = BadBroyden(self.easy_problem)
        optimum, optimum_value = gb_solver.solve()
        assert_almost_equal(self.easy_solution, optimum, decimal=5)

    def test_DFP(self):
        gb_solver = DFP(self.easy_problem)
        optimum, optimum_value = gb_solver.solve()
        assert_almost_equal(self.easy_solution, optimum, decimal=5)

    def test_BFGS(self):
        gb_solver = BFGS(self.easy_problem)
        optimum, optimum_value = gb_solver.solve()
        assert_almost_equal(self.easy_solution, optimum, decimal=5)


if __name__ == "__main__":
    unittest.main()