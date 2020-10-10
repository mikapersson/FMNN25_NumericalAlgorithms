from MPI_Communication import *
from mpi4py import MPI
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import cm
from mpl_toolkits import mplot3d


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
    elif rank == 3:
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
        temperature[4:, 0:4] = omega_1[:, :-1]
        temperature[:, 4:9] = omega_2
        temperature[:5, 9:] = omega_3[:, 1:]

        # 3D plot
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.contour3D(X, Y, temperature.T, 50, cmap='binary')        # contour
        # ax.plot_surface(X, Y, temperature.T, rstride=1, cstride=1,  # surface
                        #cmap='viridis', edgecolor='none')

        # Heatmap
        ax = sb.heatmap(temperature, cmap='YlOrBr')
        plt.title("Heat distribution of apartment")
        ax.set_axis_off()

        plt.show()
    else:
        raise ValueError("Wrong number of processes, only 4 is valid")


def main():
    plot_temperature()


main()

