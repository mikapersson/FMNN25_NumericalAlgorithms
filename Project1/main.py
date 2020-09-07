from CubicSpline import *
from basis_function import basis_function


def test_spline(control, knots):
    """
    Create and plot the spline according to the given control points and knots
    :param control: (array)
    :param knots: (array)
    :return: None
    """

    cubicspline = CubicSpline(control, knots)
    cubicspline.plot()


def test_basis_functions(knots):
    """
    Plot every basis function according to the given knots
    :param knots: (array)
    :return: None
    """

    size = 1000
    last_knot = knots[-1]
    xspace = linspace(0, last_knot, size)

    for j in range(len(knots)):
        temp_base = basis_function(size, knots, j)
        plt.plot(xspace, temp_base)

    plt.title("Basis functions from {} to {}".format("$u_0$", "$u_{}$".format({len(knots)-1})))
    plt.xlabel("u")
    plt.grid()
    plt.show()


def main():
    # We have two test sets

    test_control = array(
        [[-17, 9],
         [-40, 20],
         [-20, 0],
         [-100, -15],
         [-22, -62],
         [8, -78],
         [57, -30],
         [15, 8],
         [18, -3],
         [40, 17]])
    test_knots = array([0, 1, 2, 3, 4, 5, 6, 9, 11, 12])

    CONTROL = [(-12.73564, 9.03455),
               (-26.77725, 15.89208),
               (-42.12487, 20.57261),
               (-15.34799, 4.57169),
               (-31.72987, 6.85753),
               (-49.14568, 6.85754),
               (-38.09753, -1e-05),
               (-67.92234, -11.10268),
               (-89.47453, -33.30804),
               (-21.44344, -22.31416),
               (-32.16513, -53.33632),
               (-32.16511, -93.06657),
               (-2e-05, -39.83887),
               (10.72167, -70.86103),
               (32.16511, -93.06658),
               (21.55219, -22.31397),
               (51.377, -33.47106),
               (89.47453, -33.47131),
               (15.89191, 0.00025),
               (30.9676, 1.95954),
               (45.22709, 5.87789),
               (14.36797, 3.91883),
               (27.59321, 9.68786),
               (39.67575, 17.30712)]
    KNOTS = linspace(0, 1, 26)
    KNOTS[1] = KNOTS[2] = KNOTS[0]
    KNOTS[-3] = KNOTS[-2] = KNOTS[-1]

    test_spline(CONTROL, KNOTS)
    test_basis_functions(KNOTS)


main()


