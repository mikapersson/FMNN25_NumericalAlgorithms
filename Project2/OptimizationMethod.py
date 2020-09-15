from numpy import *


class OptimizationMethod:
    """

    """
    def __init__(self):
        self.alpha = 0.0001  # step size
        self.f = function  # object function
        self.n = 2  # the dimension of the domain, R^n

    def gradient(self, x):
        """
        Computes the gradient of f at the given point x by forward-difference (p.195 Nocedal, Wright)
        :return: (array)
        """
        g = empty((self.n, 1))
        for i in range(self.n):     # for every coordinate of x
            e = zeros((self.n, 1))  # unit vectors in the domain
            e[i] = self.alpha       # we want to take a step in the i:th direction
            g[i] = (self.f(x + self.alpha*e) - self.f(x)) / self.alpha  # (8.1) at p.195

        return g

    def hessian(self):

