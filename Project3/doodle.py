from numpy import *
from scipy.sparse import diags
from scipy.sparse.linalg import cg, spsolve


n = 3
Gamma_H = 40
Gamma_N = 15
h = 1/(n+2)


def B_vector(nx, ny, border):
    B = transpose(zeros(nx * ny))

    # Top boundary values
    for i in range(1, nx):
        B[i * ny - 1] += -Gamma_N

    # Bottom boundary. Every ny:th element is set to bottom value
    for j in range(0, nx):
        B[j * ny] += -Gamma_N

    # Left boundary. First Ny values
    B[0:ny] += -Gamma_H

    # Right Boundary. Last Ny values
    B[0:ny] += -border

    B = B / h ** 2
    return B


nx = n + 1
ny = n
Initial_T = 15

border = ones(n)*Initial_T

A = 1 / (h ** 2) * diags([-4, 1, 1, 1, 1], [0, 1, -1, ny, -ny], shape=(nx * ny, nx * ny)).toarray()
for i in range(1, nx):
    A[nx, nx - 1] = 0
    A[nx - 1, nx] = 0
#print(A)

B = B_vector(nx, ny, border)

solution = spsolve(A, B)
Gamma1 = solution[(nx - 1) * ny:]
print(solution)
print(Gamma1)