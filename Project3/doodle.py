from numpy import *
from scipy.sparse import diags
from scipy.sparse.linalg import cg, spsolve


a = eye(3)
print(a)
a = flip(a, 0)
print(a)