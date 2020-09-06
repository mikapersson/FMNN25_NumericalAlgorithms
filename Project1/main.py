from Cubic_Spline import *
from basis_function import basis_function


def main():
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
    test_knots = array([0, 1, 2, 4, 5, 6, 9, 11, 12])
    cubsplin = Cubic_Spline(test_control, test_knots)
    cubsplin.plot()

    # b = basis_function(test_knots, 10)


main()


