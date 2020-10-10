from numpy import*
from scipy import*
from scipy.sparse import diags
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg, spsolve


class heat_transfer:
    """Class for computing temperature distributions and storing the current distributions."""

    def __init__(self):

        #Boundary conditions
        self.Gamma_H = 40
        self.Gamma_WF = 5
        self.Gamma_N = 10
        self.Initial_T = 15

        #Number of (inner) grid points per unit length
        self.n = 4
        self.h = 1/self.n

        # Grid temperatures for the domains
        # Domain 1
        self.T_domain_one = ones((self.n+2,self.n+2)) * self.Initial_T
        self.T_domain_one[-1, :] = self.Gamma_N
        self.T_domain_one[0, :] = self.Gamma_N
        self.T_domain_one[0:self.n+2, 0] = self.Gamma_H

        #Nodal temperatures on the border gamma between omega 1 and omega 2
        self.Gamma1 = ones(self.n)*self.Initial_T

        # Domain 2
        self.T_domain_two = ones((2*self.n+2, self.n+2))
        self.T_domain_two[0:int(self.n*2 / 2 + 1), 0] = self.Initial_T
        self.T_domain_two[int(self.n*2 / 2 + 1):, 0] = self.Gamma_N
        self.T_domain_two[0:int(self.n*2 / 2 + 1), -1] = self.Gamma_N
        self.T_domain_two[int(self.n*2 / 2 + 1):, -1] = self.Initial_T
        self.T_domain_two[-1, :] = self.Gamma_H
        self.T_domain_two[0, :] = self.Gamma_WF

        # Inner borders toward omega one and omega three
        self.omega_one_border = ones(self.n)*self.Initial_T
        self.omega_three_border = ones(self.n)*self.Initial_T

        #Domain 3
        self.T_domain_three = ones((self.n+2,self.n+2)) * self.Initial_T
        self.T_domain_three[-1, :] = self.Gamma_N
        self.T_domain_three[0, :] = self.Gamma_N
        self.T_domain_three[0:self.n + 2, -1] = self.Gamma_H

        # Nodal temperatures on the border gamma between omega 3 and omega 2
        self.Gamma3 = ones(self.n) * self.Initial_T

    def A_matrix(self,nx,ny):
        """
        Discretization of the Laplace operator
        :param nx: Number of x-axis grid points
        :param ny: Number of y-grid points
        :return: Matrix A
        """
        A = 1 / (self.h ** 2) * diags([-4, 1, 1, 1, 1], [0, 1, -1, ny, -ny], shape=(nx * ny, nx * ny)).toarray()
        for i in range(1, nx):
            A[nx, nx - 1] = 0
            A[nx - 1, nx] = 0
        A = csr_matrix(A)
        return A

    def B_vector(self, nx, ny, domain = ""):
        """
        Array B for storing boundary conditions
        :param nx: Number of x-axis grid points
        :param ny: Number of y-axis grid points
        :param domain: Domain 1, 2 or 3 in which the temperature is evaluated.
        :return: Array B
        """

        #B is always an array of size nx*ny
        B = transpose(zeros(nx * ny))

        # Top boundary values.
        if domain == 2:
            for i in range(1, nx):
                B[i * ny - 1] += -self.Gamma_H
        if domain == 1 or domain == 3:
            for i in range(1, nx):
                B[i * ny - 1] += -self.Gamma_N

        # Bottom boundary. Every ny:th element is set to bottom value
        if domain == 2:
            for j in range(0, nx):
                B[j * ny] += -self.Gamma_WF
        if domain == 1 or domain == 3:
            for j in range(0, nx):
                B[j * ny] += -self.Gamma_N

        # Left boundary. First Ny values
        if domain == 2:
            B[0:int(ny / 2)] += -self.T_domain_one[1:self.n+1,-1]
            B[int(ny / 2):ny] += -self.Gamma_N
        if domain == 1:
            B[0:ny] += -self.Gamma_H
        if domain == 3:
            B[0:ny] += -self.omega_three_border

        # Right Boundary. Last Ny values
        if domain == 2:
            B[(nx - 1) * ny: int((nx - 1) * ny + ((nx * ny - 1) - (nx - 1) * ny) / 2) + 1] += -self.Gamma_N
            B[(nx - 1) * ny + int(((nx * ny - 1) - (nx - 1) * ny) / 2) + 1: nx * ny] += -self.T_domain_three[1:self.n+1,0]
        if domain == 3:
            B[(nx - 1) * ny: nx*ny] = -self.Gamma_H
        if domain == 1:
            B[0:ny] += -self.omega_one_border

        B = B / self.h ** 2
        return B

    def compute_distribution(self, domain = ""):

        if domain == 2:
            # Number of grid points along each axis
            nx = self.n
            ny = self.n * 2

            X_length = 1
            Y_length = 2

            # Matrix A and array b that fulfil Ax = b where x is the temperature matrix
            B = self.B_vector(nx, ny, domain = 2)
            A = self.A_matrix(nx, ny)
            solution = spsolve(A, B)

            # First ny/2 elements are the inner borders that should be sent to Omega1
            self.omega_one_border = solution[0:int(ny / 2)]

            # Last ny/2 elements are the inner borders that should be sent to Omega3
            self.omega_three_border = solution[nx * ny - int(ny / 2):]

            T = ones((ny + 2, nx + 2))
            T = self.T_domain_two

            for j in range(1, ny + 1):
                for i in range(1, nx + 1):
                    T[j, i] = solution[j + (i - 1) * ny - 1]

            self.T_domain_two = T

            return self.omega_one_border, self.omega_three_border

        if domain == 1:

            nx = self.n + 1
            ny = self.n

            A = self.A_matrix(nx, ny)
            B = self.B_vector(nx, ny, domain=1)
            solution = spsolve(A, B)

            self.Gamma1 = solution[(nx-1)*ny:]
            self.T_domain_two[1:int(self.n * 2 / 2 + 1), 0] = self.Gamma1

            T = self.T_domain_one
            for j in range(1, ny + 1):
                for i in range(1, nx + 1):
                    T[j, i] = solution[j + (i - 1) * ny - 1]
            self.T_domain_one = T
            return self.Gamma1

        if domain == 3:

            nx = self.n + 1
            ny = self.n

            A = self.A_matrix(nx, ny)
            B = self.B_vector(nx, ny, domain=3)
            solution = spsolve(A, B)

            self.Gamma3 = solution[0:ny]
            self.T_domain_two[int(self.n * 2 / 2 +1):-1, -1] = self.Gamma3
            print(self.Gamma3)


            T = self.T_domain_three
            for j in range(1, ny + 1):
                for i in range(1, nx +1):
                    T[j, i-1] = solution[j + (i - 1) * ny - 1]
            self.T_domain_three = T
            return self.Gamma3

"""
         x = linspace(0, X_length, nx + 2)
            y = linspace(0, Y_length, ny + 2)
            X, Y = meshgrid(x, y)

            # Set the imposed boundary values
            # T = zeros_like(X)
            # T = ones((ny+2,nx+2))
            T = self.T_domain_two
            
            h = plt.contourf(X, Y, T)
            plt.colorbar(aspect=4)
            plt.show()"""

            
            



















