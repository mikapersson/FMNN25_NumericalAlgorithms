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
        H_k = H_km1 + outer(delta_k - H_km1@gamma_k, H_km1@gamma_k) / dot((H_km1@delta_k).T, gamma_k)

        self.inverted_hessian = H_k


class BadBroyden(QuasiNewton):  # 3.18
    def update_hessian(self, x_next, x_k):
        delta_k = x_next - x_k
        gamma_k = self.gradient(x_next) - self.gradient(x_k)
        H_km1 = self.inverted_hessian  # "H k minus 1"
        H_k = H_km1 + outer((delta_k- H_km1@gamma_k)/inner(gamma_k, gamma_k), gamma_k)  # SOMETHING WRONG HERE

        self.inverted_hessian = H_k


class SymmetricBroyden(QuasiNewton):  # 3.20
    def update_hessian(self, x_next, x_k):
        delta_k = x_next - x_k
        gamma_k = self.gradient(x_next) - self.gradient(x_k)
        H_km1 = self.inverted_hessian  # "H k minus 1"
        u = delta_k - H_km1@gamma_k
        a = 1 / inner(u, gamma_k)
        H_k = H_km1 + a*outer(u, u)

        self.inverted_hessian = H_k


class DFP(QuasiNewton):  # 3.21
    def update_hessian(self, x_next, x_k):
        delta_k = x_next - x_k
        gamma_k = self.gradient(x_next) - self.gradient(x_k)
        H_km1 = self.inverted_hessian  # "H k minus 1"
        H_k = H_km1 + outer(delta_k, delta_k)/inner(delta_k, gamma_k) - (H_km1@gamma_k@inner(gamma_k, H_km1))/(inner(gamma_k, H_km1)@gamma_k)

        self.inverted_hessian = H_k


class BFGS(QuasiNewton):  # 3.22
    def update_hessian(self, x_next, x_k):
        delta_k = x_next - x_k
        gamma_k = self.gradient(x_next) - self.gradient(x_k)
        H_km1 = self.inverted_hessian  # "H k minus 1"
        H_k = H_km1 + (1 + (inner(gamma_k, H_km1)@gamma_k)/inner(delta_k, gamma_k))*outer(delta_k, delta_k)/inner(delta_k, gamma_k) - \
            (outer(delta_k, gamma_k)@H_km1 + H_km1@outer(gamma_k, delta_k))/inner(delta_k, gamma_k)

        self.inverted_hessian = H_k

