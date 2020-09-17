from Newton import *


class QuasiNewton(Newton):
    """
    Subclass to Newton, collecting the common methods for the deriving QuasiNewton methods (below this class)
    """

    def __init__(self, problem, lsm="inexact"):
        super().__init__(problem, lsm)

    def update_hessian(self, x_next, x_k):
        """
        Abstract method for updating self.hessian
        :param x_next: (array) to
        :param x_k: (array) from
        :return: None
        """

    def newstep(self, x):
        s = self.step_direction(x)
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
        H_km1 = self.inverted_hessian  # "H k minus 1"
        H_km1 = H_km1 + outer(delta_k - H_km1@gamma_k, H_km1@gamma_k) / dot((H_km1@delta_k).T, gamma_k)

        self.inverted_hessian = H_km1


"""
class BadBroyden(QuasiNewton):  # 3.18

class SymmetricBroyden(QuasiNewton):  # 3.19-3.20

class DFP(QuasiNewton):  # 3.21

class BFGS(QuasiNewton):  # 3.22
"""
