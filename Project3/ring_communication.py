from mpi4py import MPI

"""
Sum the ranks of each process by letting the processes 
communicate as in a 1d torus
"""

comm = MPI.COMM_WORLD
rank = comm.rank  # rank of current process
size = comm.size  # number of working processes

if rank == 0:
    print("We are working on {} processes".format(size))
    comm.send(rank, dest=1)

    s = comm.recv(source=size-1)
    print("Sum of ranks is {}".format(s))
else:
    received = comm.recv(source=rank-1)
    comm.send(received + rank, dest=(rank+1) % size)


""" FACIT
s = comm.allreduce(rank, op=MPI.SUM)
if rank == 0:
    print(s)
"""
