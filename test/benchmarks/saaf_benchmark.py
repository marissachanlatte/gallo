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

# @filename_to_problem
# def test_problem(problem):
#     source = 10*np.ones(problem.n_elements)
#     for g in range(problem.num_groups):
#         print("Starting Group ", g)
#         scalar_flux, ang_fluxes = problem.op.solve(source, "eigenvalue", g, "vacuum", tol=1e-3)
#         for i in range(4):
#             plot(problem.grid, ang_fluxes[i], problem.filename + "_ang" + str(i) + "_group" + str(g), mesh_plot=True)
#         plot(problem.grid, scalar_flux, problem.filename + "_scalar_flux" + "_group" + str(g), mesh_plot=True)
#         print("Finished Group ", g)
#     print("Max Flux: ", np.max(scalar_flux))

@filename_to_problem
def test_problem(problem):
    source = 10*np.ones(problem.n_elements)
    phis, angs = problem.op.solve_outer(source)

    # Plot Everything
    for g in range(problem.num_groups):
        scalar_flux = phis[g]
        plot(problem.grid, scalar_flux, problem.filename + "_scalar_flux" + "_group" + str(g), mesh_plot=True)
        ang_fluxes = angs[g]
        for i in range(4):
            plot(problem.grid, ang_fluxes[i], problem.filename + "_ang" + str(i) + "_group" + str(g), mesh_plot=True)

@filename_to_problem
def test_1d(problem):
    source = 10*np.ones(problem.n_elements)
    for g in range(problem.num_groups):
        scalar_flux, ang_fluxes = problem.op.solve(source, "eigenvalue", g, "vacuum", tol=1e-3)
        plot1d(scalar_flux, problem.filename + "_scalar_flux_1d", 0)

@filename_to_problem
def make_lhs(problem):
    source = np.ones(problem.n_elements)
    A = problem.op.make_lhs([.5773503, -.5773503], 0)
    print(A)

def plot1d(sol, filename, y):
    if y==0.125:
        nodes = np.array([0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1])
        flux = np.array([sol[72], sol[33], sol[62], sol[32], sol[48], sol[28], sol[56], sol[36], sol[77]])
        i = 2
    if y==0.0625:
        nodes = np.array([0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375])
        flux = np.array([sol[118], sol[117], sol[138], sol[89], sol[137], sol[112], sol[96], sol[124]])
        i = 1
    if y==0:
        nodes = np.array([0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1])
        flux = np.array([sol[0], sol[71], sol[17], sol[49], sol[8], sol[69], sol[24], sol[78], sol[1]])
        i=0
    plt.plot(nodes, flux, marker="o")
    plt.title("Scalar Flux Line Out at y=" + str(y))
    plt.savefig(filename + str(i))
    plt.clf()
    plt.close()

@filename_to_problem
def plot_mats(problem):
    plot_mesh(problem.grid, problem.mats, 'meshplot')

problem = to_problem("symmetric_fine")
print("Scattering XS: ", problem.mats.get_sigs(0, 0))
print("Total XS: ", problem.mats.get_sigt(0, 0))
#plot1d(problem.filename)
#test_1d(problem.filename)
#make_lhs(problem.filename)
test_problem(problem.filename)
#plot_mats(problem.filename)
