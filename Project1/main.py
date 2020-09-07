from CubicSpline import *
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
    test_knots = array([0, 1, 2, 3, 4, 5, 6, 9, 11, 12])
    cubsplin = CubicSpline(test_control, test_knots)
    #cubsplin.plot()


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


    #cubsplin = CubicSpline(CONTROL, KNOTS)
    # cubsplin.plot()
    # print(cubsplin(0.2))

    size = 1000
    xspace = linspace(0, 8, size)
    nvals = zeros(size)
    nvals2 = zeros(size)
    maxval = amax(test_knots)
    for i in range(0, size):
        nvals[i] = basis_function(test_knots, 0, 3, i * 8 / size)
        nvals2[i] = basis_function(test_knots, 1, 3, i * 8 / size)

    print(str(nvals.shape))
    print(str(xspace.shape))

    plt.plot(xspace, nvals, 'b')
    plt.plot(xspace, nvals2, 'r')
    plt.grid()
    plt.show()


main()


