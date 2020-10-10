from mpi4py import MPI
from heat_transfer import *
from Domain_One import*
from Domain_Two import*
from Domain_Three import*


class MPI_heat_transfer:

        comm = MPI.COMM_WORLD
        rank = comm.rank  # rank of current process
        size = comm.size  # number of working processes
        maxiteration = 3

        # Initializing domains for storing temperatures.
        print(rank)
        if rank == 0:
            Domain_One = Domain_One()
            n = Domain_One.n
        if rank == 1:
            Domain_Two = Domain_Two()
        if rank == 2:
            Domain_Three = Domain_Three()

        k = 0

        while k < maxiteration:  # Start iteration
            print("Process {}, iteration {}".format(rank+1, k+1))
            if rank == 0:
                if k == 0:
                    border = Domain_One.compute_distribution(None)
                    comm.send(border, dest=1)
                    k += 1
                else:
                    gamma = comm.recv(source=1)
                    border = Domain_One.compute_distribution(gamma)
                    comm.send(1,dest = 2)
                    comm.send(border, dest=1)
                    k+=1
                    print(Domain_One.T_domain_one)
                    print()
                    #if k == maxiteration:
                     #   print(Domain_One.T_domain_one)
                      #  print()

            if rank == 1:

                border1 = comm.recv(source=0)
                border2 = comm.recv(source=2)
                gamma1, gamma2 = Domain_Two.compute_distribution(border1, border2)
                comm.send(gamma1, dest=0)
                comm.send(gamma2, dest=2)
                k+=1
                print(Domain_Two.T_domain_two)
                print()
                # if k == maxiteration:

            if rank == 2:
                if k == 0:
                    border = Domain_Three.compute_distribution(None)
                    comm.send(border, dest=1)
                    k += 1
                else:
                    gamma = comm.recv(source=1)
                    border = Domain_Three.compute_distribution(gamma)
                    one = comm.recv(source = 0)
                    comm.send(border, dest=1)
                    k+=1
                    print(Domain_Three.T_domain_three)
                    print()
                    #if k == maxiteration:
