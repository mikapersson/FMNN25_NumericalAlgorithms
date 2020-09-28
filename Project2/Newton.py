"""Generic class for QuasiNewton"""
from numpy import *
from linesearchmethods import exact_linesearch, inexact_linesearch

"""
We refer to the Lecture03 slides when we write "section (X.Y)"
"""


class Newton:

    def __init__(self, problem, lsm="inexact", hessians="off"):
        self.epsilon = 0.000001                       # step size for approximating derivatives
        self.f = problem.function                     # object function
        self.n = problem.dimension                    # the dimension of the domain, R^n
        self.alpha = 1                                # step size in the Newton Direction
        self.values = array([])                       # the values we obtain when iterating to the optimum solution
        self.TOL = 1.e-5                              # values under TOL are set to 0
        self.start = ones((self.n, 1)) * 4            # where we start our iteration/algorithm
        self.hessian = self.compute_hessian(self.start)   # current Hessian matrix (G)
        self.inverted_hessian = linalg.inv(self.hessian)  # current inverted Hessian matrix (H)
        self.hessians = hessians
        self.all_hessians = [self.inverted_hessian]
        self.stepcoordinates = [self.start]

        # Decides which gradient-function to go after
        if problem.gradient:  # if user supplied problem instance with a gradient
            self.gradient = problem.gradient
        else:  # use default otherwise (implemented in this class)
            self.gradient = self.compute_gradient

        # Determining LSM
        valid_lsm = ["inexact", "exact"]  # valid line search methods
        if lsm not in valid_lsm:
            raise ValueError("Provided LSM \'{}\' does not exist, choose between \'inexact\' and \'exact\'".format(lsm))
        elif lsm == "inexact":
            self.lsm = lsm

            # Parameters for the inexact line search method (slide 3.12 in Lecture03)
            self.rho = 0.1
            self.sigma = 0.7
            self.tau = 0.1
            self.chi = 9

        elif lsm == "exact":
            self.lsm = lsm

    def compute_gradient(self, x):
        """
        Computes the gradient of f at the given point x by central-difference ((8.7) p.196 Nocedal, Wright)
        :return: (array)
        """
        g = empty((self.n, 1))
        for i in range(self.n):     # for every coordinate of x
            e = zeros((self.n, 1))  # unit vectors in the domain
            e[i] = self.epsilon     # we want to take a step in the i:th direction
            g[i] = (self.f(x + e) - self.f(x - e)) / (2 * self.epsilon)  # (8.1) at p.195
        return g

    def compute_hessian(self, x):
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

    def update_hessian(self, x_next):
        """
        Updates the Hessian matrix and its inverse
        :param x_next: (array) next point
        :return: None
        """

        self.hessian = self.compute_hessian(x_next)
        self.inverted_hessian = linalg.inv(self.hessian)

    def step_direction(self, x):
        """
        Computes the step direction, usually denoted 's'
        :param x: (array)
        :return: (array)
        """

        return - self.inverted_hessian@self.gradient(x)

    def newstep(self, x):
        """
        Computes coordinates for the next step in accordance with the Newton procedure.
        :return: (array)
        """

        s = self.step_direction(x)  # Newton/step direction
        new_coordinates = x + self.alpha * s
        self.update_hessian(new_coordinates)
        return new_coordinates

    def linesearch(self, x):
        """
        Solving the sub-problem of finding how far we take a step in the direction 's' computed below
        :param x:
        :return:
        """

        object_function = self.f
        s = self.step_direction(x)
        if self.lsm == 'inexact':
            return inexact_linesearch(object_function, x, s, self.rho, self.sigma, self.tau, self.chi)
        elif self.lsm == 'exact':
            return exact_linesearch(object_function, x, s)

    def newHessian(self):
        """Returns Hessian. Overridden in 9 special methods"""
        raise NotImplementedError("Can only be called through inherited classes")

    def termination_criterion(self, x):
        """
        Asserts that the criterions for optimum are fulfilled. The criterions are:
        1. Hessian is symmetric and positive definite.
        2. The gradient is zero.
        :return: (boolean) that is true if the criteria are fulfilled
        """

        Hess = self.hessian
        if not all(abs(self.gradient(x)) < self.TOL):
            return False
        else:
            try:
                linalg.cholesky(Hess)
                #print("POSITIVE DEFINITE AND SYMMETRIC")
            except:
                return False  # If the Choleskymethod gives error False is returned.

        print("SOLVED!\nNumber of iterations: {}".format(len(self.values[0])))
        return True

    def solve(self):
        """

        :return: (list)
        """

        x = self.start
        solved = self.termination_criterion(x)
        value = x
        self.values = value
        while solved is False:
            self.alpha = self.linesearch(value)
            newvalue = self.newstep(value)
            value = newvalue
            solved = self.termination_criterion(value)
            self.values = hstack([self.values, value])

        if self.hessians == "on":
            return self.all_hessians, self.stepcoordinates

        return [value, self.f(value)]

