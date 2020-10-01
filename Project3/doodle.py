from mpi4py import MPI
from numpy import array, random


comm = MPI.COMM_WORLD

sendbuf = []
rank = comm.rank
size = comm.size

if rank == 0:
    m = [i for i in range(size)]
    print("Original array on rank 0:\n{}".format(m))
    sendbuf = m

v = comm.scatter(sendbuf, root=0)
print("I (proc. {}) got this array:\n{}".format(rank, v))
