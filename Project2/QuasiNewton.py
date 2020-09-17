"""Generic class for QuasiNewton"""
from numpy import *
from math import inf
from linesearchmethods import exact_linesearch, inexact_linesearch


class QuasiNewton:

    def __init__(self, problem, lsm="inexact"):
        self.epsilon = 0.0001      # step size
        self.f = problem.function  # object function
        self.n = 2                 # the dimension of the domain, R^n
        self.alpha = 1             #
        self.values = array([])    #
        self.TOL = 1.e-8           # tolerance for the 0-element

        # Determining LSM
        valid_lsm = ["inexact", "exact"]  # valid line search methods
        if lsm not in valid_lsm:
            raise ValueError("Provided LSM \'{}\' does not exist, choose between \'inexact\' or \'exact\'".format(lsm))
        elif lsm == "inexact":
            self.lsm = lsm

            # PARAMETERS FOR INEXACT LINE SEARCH METHOD (slide 3.12 in Lecture03)
            self.rho = 0.1
            self.sigma = 0.7
            self.tau = 0.1
            self.chi = 9

        elif lsm == "exact":
            self.lsm = lsm

    def gradient(self, x):
        """
        Computes the gradient of f at the given point x by central-difference ((8.7) p.196 Nocedal, Wright)
        :return: (array)
        """
        g = empty((self.n, 1))
        for i in range(self.n):  # for every coordinate of x
            e = zeros((self.n, 1))  # unit vectors in the domain
            e[i] = self.epsilon  # we want to take a step in the i:th direction
            g[i] = (self.f(x + e) - self.f(x - e)) / (2 * self.epsilon)  # (8.1) at p.195
        return g

    def hessian(self, x):
        """
        Computes the inverse hessian of f at the given point x by forward-difference (p.201 Nocedal, Wright)
        :return: nxn matrix
        """
        G = empty((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                direction1 = zeros((self.n, 1))
                direction2 = zeros((self.n, 1))
                direction1[i] = self.epsilon
                direction2[j] = self.epsilon
                G[i, j] = (self.f(x + direction1 + direction2) - self.f(x + direction2) - self.f(
                    x + direction1) + self.f(x)) / (self.epsilon ** 2)
        G = (G + G.transpose()) / 2
        return G

    def newton_direction(self, x):
        """

        :param x:
        :return:
        """

        inverse_hessian = linalg.inv(self.hessian(x))
        g = self.gradient(x)
        newton_direction = -inverse_hessian.dot(g)
        return newton_direction

    def newstep(self, x):
        """
        Computes coordinates for the next step in accordance with the Quasi Newton procedure.
        :return: (array)
        """

        newton_direction = self.newton_direction(x)  # The Newton direction determines step direction.
        new_coordinates = x + self.alpha * newton_direction

        return new_coordinates

    def linesearch(self, x):
        if self.lsm == 'inexact':
            return inexact_linesearch(self.f, x, self.newton_direction(x), self.rho, self.sigma, self.tau, self.chi)
        elif self.lsm == 'exact':
            return exact_linesearch(self.f, x, self.newton_direction(x))

    def newHessian(self):
        """Returns Hessian. Overridden in 9 special methods"""
        return

    def termination_criterion(self, x):
        """
        Asserts that the criterions for optimum are fulfilled. The criterions are:
        1.Hessian is symmetric and positive definite.
        2. The gradient is zero.
        :return: boolean that is true if criteria are fulfilled
        """
        Hess = self.hessian(x)
        if not all(self.gradient(x) < self.TOL):
            return False
        """
        else:
            try:
                linalg.cholesky(Hess)
            except:
                return False  # If the Choleskymethod gives error False is returned.
        """
        print("SOLVED!")
        return True

    def solve(self):
        x = ones((self.n, 1)) * 2
        self.alpha = self.linesearch(x)
        solved = self.termination_criterion(x)
        value = x
        self.values = value
        while solved is False:
            print(self.alpha)
            self.alpha = self.linesearch(x)
            newvalue = self.newstep(value)
            value = newvalue
            solved = self.termination_criterion(value)
            self.values = hstack([self.values, value])
            print(self.alpha)
        return [value, self.f(value)]
