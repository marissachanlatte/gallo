import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import sys
sys.path.append('../../src')

from formulations.nda import NDA
from formulations.saaf import SAAF
from fe import *
from materials import Materials
from problem import Problem
from plot import plot
from solvers import Solver

def to_problem(mesh, mat, filename):
    nodefile = "../test_inputs/" + mesh + ".node"
    elefile = "../test_inputs/" + mesh + ".ele"
    matfile = "../test_inputs/" + mat + ".mat"
    grid = FEGrid(nodefile, elefile)
    mats = Materials(matfile)
    ho = SAAF(grid, mats)
    ho_solver = Solver(ho)
    n_elements = grid.get_num_elts()
    num_groups = mats.get_num_groups()
    source = np.ones((num_groups, n_elements))
    op = NDA(grid, mats, ho_solver, source)
    solver = Solver(op)

    return Problem(op=op, mats=mats, grid=grid, solver=solver, filename=filename)

def filename_to_problem(func):
    def _filename_to_problem(mesh, mat, filename):
        return func(problem=to_problem(mesh, mat, filename))
    return _filename_to_problem

@filename_to_problem
def test_problem(problem):
    source = np.ones((problem.num_groups, problem.n_elements))
    #phis, angs, eigenvalue = problem.op.solve(source, eigenvalue=True)
    phis = problem.solver.solve(source, eigenvalue=False)

    # Plot Everything
    for g in range(problem.num_groups):
        scalar_flux = phis[g]
        plot(problem.grid, scalar_flux, problem.filename + "_scalar_flux" + "_group" + str(g))

test_problem("symmetric_fine", "scattering2g", "test")
