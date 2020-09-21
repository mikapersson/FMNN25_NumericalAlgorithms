from OptimizationProblem import OptimizationProblem
from Newton import *
from QuasiNewton import *
from linesearchmethods import *
from main import rosenbrock
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

    f = rosenbrock
    problem = OptimizationProblem(f)
    solution = array([[1],
                      [1]])
    TOL = 1.e-5

    def test_gradient(self):
        pass

    def test_termination_criterion(self):
        pass

    def test_newton(self):
        newton_solver = Newton(self.problem)
        optimum, optimum_value = newton_solver.solve()
        assert_almost_equal(self.solution, optimum, decimal=5)

    def test_good_broyden(self):
        pass

    def test_bad_broyden(self):
        pass

    def test_DFP(self):
        pass

    def test_BFGS(self):
        pass


if __name__ == "__main__":
    unittest.main()