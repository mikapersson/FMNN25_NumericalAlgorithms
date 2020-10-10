from MPI_Communication import *
import matplotlib.pyplot as plt
from matplotlib import cm


"""
File for plotting the temperature distribution of the apartment
"""


def plot_temperature():
    comm = MPI.COMM_WORLD
    rank = comm.rank  # rank of current process

    # Calculate temperature distribution
    if rank == 0:
        omega_1 = solve_heat_transfer(comm)
        comm.send(omega_1, dest=3)
    elif rank == 1:
        omega_2 = solve_heat_transfer(comm)
        comm.send(omega_2, dest=3)
    elif rank == 2:
        omega_3 = solve_heat_transfer(comm)
        comm.send(omega_3, dest=3)
    elif rank == 4:
        omega_1 = comm.recv(source=0)
        omega_2 = comm.recv(source=1)
        omega_3 = comm.recv(source=2)

        # Construct temperature distribution matrix and plot
        x_len = 9
        y_len = 13
        x = arange(0, x_len)
        y = arange(0, y_len)
        X, Y = meshgrid(x, y)
        temperature = zeros((x_len, y_len))
        temperature[4:, 0:5] = omega_1
        temperature[:, 5:10] = omega_2
        temperature[:5, 10:] = omega_3

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y ,temperature, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()
    else:
        raise ValueError("Wrong number of processes, only 4 is valid")


def main():
    plot_temperature()


main()

