import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
import sys
sys.path.append('../src')

from diffusion import Diffusion
from fe import *
from materials import Materials
from plot import plot

# Uniform Source Test
def source_function(x, filename):
    if filename=="uniform_source":
        return 1
    elif filename=="box_2":
        if 4 < x[0] < 6 and 4 < x[1] < 6:
            return 1
        else:
            return 0
    else:
      print("Input not supported")

def diffusion_test(filename):
    nodefile = "test_inputs/" + filename + ".node"
    elefile = "test_inputs/" + filename + ".ele"
    matfile = "test_inputs/" + filename + ".mat"
    grid = FEGrid(nodefile, elefile)
    mats = Materials(matfile)

    op = Diffusion(grid, mats)

    A = op.get_matrix()

    n_elements = grid.get_num_elts()
    source_terms = np.zeros(n_elements)
    for i in range(n_elements):
      cent = grid.centroid(i)
      source_terms[i] = source_function(cent, filename)
    rhs = op.make_rhs(source_terms)
    internal_nodes = linalg.cg(A, rhs)

    phi = reinsert(grid, internal_nodes[0])
    plot(grid, phi, filename+"_test")

#diffusion_test("uniform_source")
diffusion_test("box_2")
