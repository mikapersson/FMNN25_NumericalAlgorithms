from numpy import*
from scipy import*
from scipy.sparse import diags
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg, spsolve

class Domain_One:

    def __init__(self):
        # Boundary conditions
        self.Gamma_H = 40
        self.Gamma_WF = 5
        self.Gamma_N = 15
        self.Initial_T = 15

        # Number of (inner) grid points per unit length
        self.n = 3
        self.h = 1 / self.n

        self.omega = 0.8

        self.T_domain_one = ones((self.n + 2, self.n + 2)) * self.Initial_T
        self.T_domain_one[-1, :] = self.Gamma_N
        self.T_domain_one[0, :] = self.Gamma_N
        self.T_domain_one[0:self.n + 2, 0] = self.Gamma_H

    def A_matrix(self, nx, ny):
        """
        Discretization of the Laplace operator
        :param nx: Number of x-axis grid points
        :param ny: Number of y-grid points
        :return: Matrix A
        """
        A = 1 / (self.h ** 2) * diags([-4, 1, 1, 1, 1], [0, 1, -1, ny, -ny], shape=(nx * ny, nx * ny)).toarray()
        for i in range(1, nx):
            A[i * ny, i * (ny - 1)] = 0
            A[i * (ny - 1), i * ny] = 0
        A = csr_matrix(A)
        return A

    def B_vector(self, nx, ny, border):

        B = transpose(zeros(nx * ny))

        # Top boundary values
        for i in range(1, nx):
            B[i * ny - 1] += -self.Gamma_N

        # Bottom boundary. Every ny:th element is set to bottom value
        for j in range(0, nx):
            B[j * ny] += -self.Gamma_N

        # Left boundary. First Ny values
        B[0:ny] += -self.Gamma_H

        # Right Boundary. Last Ny values
        B[-ny:] += -border

        B = B / self.h ** 2
        return B

    def compute_distribution(self, border):

        if border is None:
            border = ones(self.n)*self.Initial_T

        nx = self.n + 1
        ny = self.n

        A = self.A_matrix(nx, ny)
        B = self.B_vector(nx, ny, border)
        solution = spsolve(A, B)

        self.Gamma1 = solution[(nx - 1) * ny:]
        #self.T_domain_two[1:int(self.n * 2 / 2 + 1), 0] = self.Gamma1

        T = self.T_domain_one
        Told = self.T_domain_one

        for j in range(1, ny + 1):
            for i in range(1, nx + 1):
                T[j, i] = solution[j + (i - 1) * ny - 1]
                #T[2,1] is solution[1]
        self.T_domain_one = T

        self.T_domain_one = self.omega*self.T_domain_one + (1-self.omega)*Told


        return self.Gamma1