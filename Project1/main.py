from Cubic_Spline import *


def main():
    test_control = array(
        [[-40, 20], [-20, 0], [-100, -15], [-22, -62], [8, -78], [57, -30], [15, 8], [18, -3], [40, 17]])
    knots = array([0, 1, 2, 4, 5])
    cubsplin = Cubic_Spline(knots)
    cubsplin.plot()

main()


