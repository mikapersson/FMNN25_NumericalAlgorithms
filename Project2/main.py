from numpy import *
from OptimizationProblem import*
from QuasiNewton import*


def testfunction(x):
    value = sin(x[0]) + sin(x[1])
    return value


def testgradient(x):
    grad = cos(x[0])+cos(x[1])
    return grad


problem = OptimizationProblem(testfunction, testgradient)
solution = QuasiNewton(problem)
a = solution.solve()
print(a)