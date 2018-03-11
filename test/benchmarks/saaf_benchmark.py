import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
from itertools import product
import sys
sys.path.append('../../src')

from formulations.saaf import SAAF
from fe import *
from materials import Materials
from problem import Problem
from plot import *

def to_problem(filename):
    nodefile = "../test_inputs/" + filename + ".node"
    elefile = "../test_inputs/" + filename + ".ele"
    matfile = "../test_inputs/" + filename + ".mat"
    grid = FEGrid(nodefile, elefile)
    mats = Materials(matfile)
    op = SAAF(grid, mats)
    n_elements = grid.get_num_elts()
    num_groups = mats.get_num_groups()
    return Problem(op=op, mats=mats, grid=grid, filename=filename)

def filename_to_problem(func):
    def _filename_to_problem(filename):
        return func(problem=to_problem(filename))
    return _filename_to_problem
   
@filename_to_problem
def test_problem(problem):
    source = 10*np.ones(problem.n_elements)
    scalar_flux, ang_fluxes = problem.op.solve(source, "eigenvalue", 0, "reflecting", tol=1e-1)
    for i in range(4):
        plot(problem.grid, ang_fluxes[i], "saaf" + str(i))
    plot(problem.grid, scalar_flux, "scalar_flux")

problem = to_problem("D.1")
print("Scattering XS: ", problem.mats.get_sigs(0, 0))
print("Total XS: ", problem.mats.get_sigt(0, 0))
test_problem(problem.filename)













