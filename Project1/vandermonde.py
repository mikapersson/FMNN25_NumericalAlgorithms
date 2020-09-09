from greville_abscissae import greville_abscissae
from basis_function import basis_function_rec
from numpy import array, zeros

"""
Authors: Mika Persson & Viktor Sambergs
"""

def vandermonde(knots):
    """
    Creates the Vandermonde matrix in slide 1.12 for interpolation purposes
    :param knots: (array)
    :return: (array)
    """

    chis = greville_abscissae(knots)
    size = len(chis)  # dimension of vandermonde matrix is (size, size) (L + 1)

    vandermonde_matrix = zeros((size, size))

    for chi_index in range(size):
        for base_index in range(size):
            temp_value = basis_function_rec(knots, base_index, 3, chis[chi_index])
            vandermonde_matrix[chi_index, base_index] = temp_value
    return vandermonde_matrix




