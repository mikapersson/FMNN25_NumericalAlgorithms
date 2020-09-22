# Its constructor should take an
# objective function as input, optionally also its gradient can be provided.

from numpy import array


class OptimizationProblem:
    """
    Defines a problem, independent of solution method.
    """

    def __init__(self, function, gradient=None, dimension=2):
        self.function = function
        self.gradient = gradient
        self.dimension = dimension
