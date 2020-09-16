"""Generic class for QuasiNewton"""
from numpy import *


class QuasiNewton:

    def __init__(self,problem):
        self.epsilon = 0.0001      # step size
        self.f = problem.function  # object function
        self.n = 2                 # the dimension of the domain, R^n
        self.alpha = 1

    def gradient(self, x):
        """
        Computes the gradient of f at the given point x by central-difference ((8.7) p.196 Nocedal, Wright)
        :return: (array)
        """
        g = empty((self.n, 1))
        for i in range(self.n):  # for every coordinate of x
                e = zeros((self.n, 1))  # unit vectors in the domain
                e[i] = self.epsilon  # we want to take a step in the i:th direction
                g[i] = (self.f(x + e) - self.f(x - e)) / (2*self.epsilon)  # (8.1) at p.195
        return g

    def hessian(self, x):
        """
        Computes the inverse hessian of f at the given point x by forward-difference (p.201 Nocedal, Wright)
        :return: nxn matrix
        """
        G = empty((self.n,self.n))
        for i in range(self.n):
            for j in range(self.n):
                direction1 = zeros((self.n,1))
                direction2 = zeros((self.n,1))
                direction1[i] = self.epsilon
                direction2[j] = self.epsilon
                G[i,j] = (self.f(x + direction1 + direction2) - self.f(x+direction2) - self.f(x+direction1) + self.f(x))/(self.epsilon**2)
        G = (G + G.transpose())/2
        return G

    def newton_direction(self,x):
        """INTE ANVÄND ÄN. VET EJ OM VI KOMMER BEHÖVA DENNA SOM METOD"""
        inverse_hessian = self.hessian(x)
        g = self.gradient(x)
        newton_direction = inverse_hessian*g
        print(newton_direction)
        return newton_direction

    def newstep(self,x):
        """
        Computes coordinates for the next step in accordance with the Quasi Newton procedure.
        :return: (array)
        """

        inverse_hessian = linalg.inv(self.hessian(x))
        print("Inverse Hess")
        print(inverse_hessian)
        g = self.gradient(x)
        print("Gradient")
        print(g)

        newton_direction = inverse_hessian.dot(g)# The Newton direction determines step direction.
        print(newton_direction)
        print("NEW")

        new_coordinates = x-newton_direction
        return new_coordinates

    def newHessian(self):
        """Returns Hessian. Overridden in 9 special methods"""
        return

    def exactlinesearch(self):
        """Defines exact line search"""
        return 1

    def inexactlinesearch(self):
        """Defines inexact linesearch"""
        return 1

    def termination_criterion(self,x):
        """
        Asserts that the criterions for optimum are fulfilled. The criterions are:
        1.Hessian is symmetric and positive definite.
        2. The gradient is zero.
        :return: boolean that is true if criteria are fulfilled
        """
        Hess = self.hessian(x)
        if self.gradient(x).all() != 0.00:
            return False
        else:
            try:
                linalg.cholesky(Hess)
            except:
                return False #If the Choleskymethod gives error False is returned.

        print("SOLVED!")
        return True

    def solve(self):
        x = zeros((2,1))
        x[0] = 1
        x[1] = 1
        solved = self.termination_criterion(x)
        value = x
        values = [value]
        while solved is False:
            newvalue = self.newstep(value)
            value = newvalue
            solved = self.termination_criterion(value)
            values = [values, value]
        return value