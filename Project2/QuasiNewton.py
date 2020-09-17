from Newton import *


class QuasiNewton(Newton):
    """
    Subclass to Newton, collecting the common methods for the deriving QuasiNewton methods (below this class)
    """

    def __init__(self, problem, lsm="inexact"):
        super().__init__(problem, lsm)
        self.hessian = self.hessian(self.start)  # current Hessian matrix (G)
        self.invhessian = invert(self.hessian)  # current inverted Hessian matrix (H)

    def update_hessian(self, x_next, x_k):
        """
        Abstract method for updating self.hessian
        :param x_next: (array) to
        :param x_k: (array) from
        :return: None
        """

    def newstep(self, x):
        s = - self.invhessian@self.gradient(x)
        new_coordinates = x + self.alpha * s
        self.update_hessian(new_coordinates, x)
        return new_coordinates


class GoodBroyden(QuasiNewton):  # 3.17
    """
    Uses simple Broyden rank-1 update of the Hessian G and applies Sherman-Morisson's formula,
    see sections 3.15-3.17
    """

    def update_hessian(self, x_next, x_k):
        delta_k = x_next - x_k
        gamma_k = self.gradient(x_next) - self.gradient(x_k)
        H_km1 = self.invhessian  # "H k minus 1"
        H_km1 = H_km1 + outer(delta_k - H_km1 * gamma_k, H_km1 * gamma_k) / \
                       inner(H_km1 * delta_k, gamma_k)
        self.invhessian = H_km1


"""
class BadBroyden(QuasiNewton):  # 3.18

class SymmetricBroyden(QuasiNewton):  # 3.19-3.20

class DFP(QuasiNewton):  # 3.21

class BFGS(QuasiNewton):  # 3.22
"""
