import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
import sys
sys.path.append('../src')

from diffusion import Diffusion
from fe import *
from materials import Materials
from plot import plot

def source_function(x):
    if 4 <= x[0] <= 6 and 4 <= x[1] <=6:
      return 1
    else:
      return 0

nodefile = "test_inputs/box.node"
elefile = "test_inputs/box.ele"
matfile = "test_inputs/box.mat"
grid = FEGrid(nodefile, elefile)
mats = Materials(matfile)

op = Diffusion(grid, mats)

A = op.get_matrix()

n_elements = grid.get_num_elts()
source_terms = np.zeros(n_elements)
for i in range(n_elements):
  cent = grid.centroid(i)
  source_terms[i] = source_function(cent)
rhs = op.make_rhs(source_terms)
internal_nodes = linalg.cg(A, rhs, tol=1e-6)

phi = reinsert(grid, internal_nodes[0])

plot(grid, phi, "diffusion_test")

