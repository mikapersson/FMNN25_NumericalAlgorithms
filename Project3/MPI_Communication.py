from mpi4py import MPI
from Domain_One import*
from Domain_Two import*
from Domain_Three import*
import sys


class MPI_heat_transfer:

        comm = MPI.COMM_WORLD
        rank = comm.rank  # rank of current process
        size = comm.size  # number of working processes

        # Initializing domains for storing temperatures.

        #print("Entered process", rank+1)

        if rank == 0:
            Domain_One = Domain_One()
        elif rank == 1:
            Domain_Two = Domain_Two()
        elif rank == 2:
            Domain_Three = Domain_Three()

        k = 0  # iteration counter
        max_iteration = int(sys.argv[-1])

        while k < max_iteration:
            print("Process {}, iteration {}".format(rank + 1, k))
            if rank == 0:
                if k == 0:
                    border = Domain_One.compute_distribution(None)
                    comm.send(border, dest=1)
                    k += 1
                else:
                    gamma = comm.recv(source = 1)
                    border = Domain_One.compute_distribution(gamma)

                    k += 1
                    if k < max_iteration:  # stop sending information to rank1
                        comm.send(border,dest=1)
                    else:
                        print(Domain_One.T_domain_one)
                    # print(border)

            if rank == 1:
                border1 = comm.recv(source=0)
                border2 = comm.recv(source=2)
                gamma1, gamma2 = Domain_Two.compute_distribution(border1, border2)
                k += 1
                comm.send(gamma1, dest=0)
                comm.send(gamma2, dest=2)

                if k == max_iteration:  # last iteration
                    print(Domain_Two.T_domain_two)

            if rank == 2:
                if k == 0:
                    border = Domain_Three.compute_distribution(None)
                    comm.send(border, dest=1)
                    k += 1
                else:
                    gamma = comm.recv(source=1)
                    border = Domain_Three.compute_distribution(gamma)

                    k += 1
                    if k < max_iteration:
                        comm.send(border, dest=1)
                    else:
                        print(Domain_Three.T_domain_three)






