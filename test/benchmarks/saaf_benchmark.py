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
    source = np.ones(problem.n_elements)
    for g in range(problem.num_groups):
        print("Starting Group ", g)
        scalar_flux, ang_fluxes = problem.op.solve(source, "eigenvalue", g, "vacuum", tol=1e-3)
        for i in range(4):
            plot(problem.grid, ang_fluxes[i], problem.filename + "_ang" + str(i) + "_group" + str(g), mesh_plot=True)
        plot(problem.grid, scalar_flux, problem.filename + "_scalar_flux" + "_group" + str(g), mesh_plot=True)
        print("Finished Group ", g)

@filename_to_problem
def make_lhs(problem):
    source = np.ones(problem.n_elements)
    A = problem.op.make_lhs([.5773503, -.5773503], 0)
    print(A)

problem = to_problem("symmetric")
print("Scattering XS: ", problem.mats.get_sigs(0, 0))
print("Total XS: ", problem.mats.get_sigt(0, 0))
test_problem(problem.filename)
#make_lhs(problem.filename)













