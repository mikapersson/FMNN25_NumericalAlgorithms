
from mpi4py import MPI
from heat_transfer import *
from Domain_One import*
from Domain_Two import*
from Domain_Three import*

def solve_heat_transfer(comm = None):

    if comm is None:
        comm = MPI.COMM_WORLD


    rank = comm.rank  # rank of current process
    size = comm.size  # number of working processes
    maxiteration = 10

    # Initializing domains for storing temperatures.
    print(rank)
    if rank == 0:
        print("Process 1")
        Domain_1 = Domain_One()
    if rank == 1:
        print("Process 2")
        Domain_2 = Domain_Two()
    if rank == 2:
        print("Process 3")
        Domain_3 = Domain_Three()

    k = 1

    while k<maxiteration:

        if rank == 0:
            if k == 1:
                border = Domain_1.compute_distribution(None)
                comm.send(border, dest=1)
                k += 1
            else:
                gamma = comm.recv(source=1)
                border = Domain_1.compute_distribution(gamma)
                comm.send(1,dest = 2)
                comm.send(border, dest=1)
                k+=1
                if k == maxiteration:
                    return Domain_1.T_domain_one
                    print(Domain_1.T_domain_one)
                    print()

        if rank == 1:

            border1 = comm.recv(source=0)
            border2 = comm.recv(source=2)
            gamma1, gamma2 = Domain_2.compute_distribution(border1, border2)
            comm.send(gamma1, dest=0)
            comm.send(gamma2, dest=2)
            k+=1
            if k == maxiteration:
                return Domain_2.T_domain_two

                print(Domain_2.T_domain_two)
                print()

        if rank == 2:
            if k == 1:
                border = Domain_3.compute_distribution(None)
                comm.send(border, dest=1)
                k += 1
            else:
                gamma = comm.recv(source=1)
                border = Domain_3.compute_distribution(gamma)
                one = comm.recv(source = 0)
                comm.send(border, dest=1)
                k+=1
                if k == maxiteration:
                    return Domain_3.T_domain_three
                    print(Domain_3.T_domain_three)
                    print()














