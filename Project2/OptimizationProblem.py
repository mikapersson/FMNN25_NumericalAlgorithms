# Its constructor should take an
# objective function as input, optionally also its gradient can be provided.


class OptimizationProblem:
    """
    Defines a problem, independent of solution method.
    """

    def __init__(self, function, gradient):
        self.function = function
        self.gradient = gradient
