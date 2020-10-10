from numpy import*
from scipy import*
from scipy.sparse import diags
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg, spsolve


class Domain_Two:

    def __init__(self, n):
        # Boundary conditions
        self.Gamma_H = 40
        self.Gamma_WF = 5
        self.Gamma_N = 15
        self.Initial_T = 15

        self.omega = 0.8  # coefficient used in relaxation (step 3 in iteration)

        # Number of (inner) grid points per unit length
        self.n = n
        self.h = 1 / self.n

        # Initialize the domain/room 'omega 2'
        self.T_domain_two = ones((2 * (self.n + 2) - 1, self.n + 2)) * self.Initial_T
        self.T_domain_two[0, :] = self.Gamma_H  # heater
        self.T_domain_two[-1, :] = self.Gamma_WF  # window

    def A_matrix(self, nx, ny):
        """
        Discretization of the Laplace operator
        :param nx: Number of x-axis grid points
        :param ny: Number of y-grid points
        :return: Matrix A
        """
        A = 1 / (self.h ** 2) * diags([-4, 1, 1, 1, 1], [0, 1, -1, ny, -ny], shape=(nx * ny, nx * ny)).toarray()
        for i in range(1, nx):
            A[i * (ny - 1), i * ny] = 0
            A[i * ny, i * (ny - 1)] = 0
        A = csr_matrix(A)
        return A

    def B_vector(self, nx, ny, Gamma_1, Gamma_3):
        """
        Array B for storing boundary conditions
        :param nx: Number of x-axis grid points
        :param ny: Number of y-axis grid points
        :return: Array B
        """

        # B is always an array of size nx*ny
        B = transpose(zeros(nx * ny))

        # Top boundary values.

        for i in range(1, nx):
            B[i * ny - 1] += -self.Gamma_H

        # Bottom boundary. Every ny:th element is set to bottom value

        for j in range(0, nx):
            B[j * ny] += -self.Gamma_WF

        # Left boundary. First Ny values
        B[0:int(ny / 2)] += -Gamma_1
        B[int(ny / 2):ny] += -self.Gamma_N

        # Right Boundary. Last Ny values
        B[(nx - 1) * ny: int((nx - 1) * ny + ((nx * ny - 1) - (nx - 1) * ny) / 2) + 1] += -self.Gamma_N
        B[(nx - 1) * ny + int(((nx * ny - 1) - (nx - 1) * ny) / 2) + 1: nx * ny] += -Gamma_3

        B = B / self.h ** 2
        return B

    def compute_distribution(self, Gamma_1, Gamma_3):
        # Number of grid points along each axis
        nx = self.n
        ny = self.n * 2

        # Matrix A and array b that fulfil Ax = b where x is the temperature matrix
        B = self.B_vector(nx, ny, Gamma_1, Gamma_3)
        A = self.A_matrix(nx, ny)
        solution = spsolve(A, B)

        # Last ny/2 elements are the inner borders that should be sent to Omega1
        omega_one_border = solution[nx * ny - int(ny / 2):]

        # First ny/2 elements are the inner borders that should be sent to Omega3
        omega_three_border = solution[0:int(ny / 2)]

        T = self.T_domain_two
        Told = self.T_domain_two

        for j in range(1, ny + 1):
            for i in range(1, nx + 1):
                T[j, i] = solution[j + (i - 1) * ny - 1]

        self.T_domain_two = T
        self.T_domain_two[-(self.n + 1):-1, 0] = Gamma_1
        self.T_domain_two[1:int(self.n * 2 / 2 + 1), -1] = Gamma_3

        self.T_domain_two = self.omega*self.T_domain_two + (1-self.omega)*Told  # relaxation

        return omega_one_border, omega_three_border
