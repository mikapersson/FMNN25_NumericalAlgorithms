from mpi4py import MPI
from heat_transfer import *
from Domain_One import*
from Domain_Two import*
from Domain_Three import*


def solve_heat_transfer(comm=None):

    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.rank  # rank of current process
    print(rank)
    n = 3  # number of inner grid points per unit length
    maxiteration = 3

    # Initializing domains for storing temperatures.
    print(rank)
    if rank == 0:
        Domain_1 = Domain_One(n)
    if rank == 1:
        Domain_2 = Domain_Two(n)
    if rank == 2:
        Domain_3 = Domain_Three(n)
    else:
        raise ValueError("Wrong number of processes, only 3 is valid")

    k = 0

    while k < maxiteration:  # Start iteration
        print("Process {}, iteration {}".format(rank+1, k+1))
        if rank == 0:
            if k == 0:
                border = Domain_1.compute_distribution(None)
                comm.send(border, dest=1)
                k += 1
            else:
                gamma = comm.recv(source=1)
                border = Domain_1.compute_distribution(gamma)
                comm.send(1,dest = 2)
                comm.send(border, dest=1)
                k += 1
                # print(Domain_1.T_domain_one)
                # print()

                if k == maxiteration:
                    return Domain_1.T_domain_one

        if rank == 1:

            border1 = comm.recv(source=0)
            border2 = comm.recv(source=2)
            gamma1, gamma2 = Domain_2.compute_distribution(border1, border2)
            comm.send(gamma1, dest=0)
            comm.send(gamma2, dest=2)
            k += 1
            # print(Domain_2.T_domain_two)
            # print()

            if k == maxiteration:
                return Domain_2.T_domain_two

        if rank == 2:
            if k == 0:
                border = Domain_3.compute_distribution(None)
                comm.send(border, dest=1)
                k += 1
            else:
                gamma = comm.recv(source=1)
                border = Domain_3.compute_distribution(gamma)
                one = comm.recv(source = 0)
                comm.send(border, dest=1)
                k+=1
                # print(Domain_3.T_domain_three)
                # print()

                if k == maxiteration:
                    return Domain_3.T_domain_three

def main():
    solve_heat_transfer()


main()