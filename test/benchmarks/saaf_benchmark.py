import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import cProfile
from itertools import product
import sys
sys.path.append('../../src')

from formulations.saaf import SAAF
from fe import *
from materials import Materials
from problem import Problem
from plot import *
from solvers import *

def to_problem(mesh, mats, filename):
    nodefile = "../test_inputs/" + mesh + ".node"
    elefile = "../test_inputs/" + mesh + ".ele"
    matfile = "../test_inputs/" + mats + ".mat"
    print(matfile)
    grid = FEGrid(nodefile, elefile)
    mats = Materials(matfile)
    op = SAAF(grid, mats)
    solver = Solver(op)
    n_elements = grid.get_num_elts()
    num_groups = mats.get_num_groups()
    return Problem(op=op, mats=mats, grid=grid, solver=solver, filename=filename)

def filename_to_problem(func):
    def _filename_to_problem(mesh, mats, filename):
        return func(problem=to_problem(mesh, mats, filename))
    return _filename_to_problem

@filename_to_problem
def test_problem(problem):
    source = np.ones((problem.num_groups, problem.n_elements))
    #phis, angs, eigenvalue = problem.solver.solve(source, eigenvalue=True)
    phis, angs = problem.solver.solve(source, eigenvalue=False)

    # Plot Everything
    for g in range(problem.num_groups):
        scalar_flux = phis[g]
        plot(problem.grid, scalar_flux, problem.filename + "_scalar_flux" + "_group" + str(g))
        ang_fluxes = angs[g]
        for i in range(4):
            plot(problem.grid, ang_fluxes[i], problem.filename + "_ang" + str(i) + "_group" + str(g))

@filename_to_problem
def test_1d(problem):
    source = np.ones(problem.n_elements)
    for g in range(problem.num_groups):
        scalar_flux, ang_fluxes = problem.op.solve(source, "eigenvalue", g, "vacuum", tol=1e-3)
        plot1d(scalar_flux, problem.filename + "_scalar_flux_1d", 0)

@filename_to_problem
def make_lhs(problem):
    source = np.ones(problem.n_elements)
    A = problem.op.make_lhs([.5773503, -.5773503], 0)
    print(A)

@filename_to_problem
def get_mat_stats(problem):
    num_mats = problem.mats.get_num_mats()
    num_groups = problem.mats.get_num_groups()
    for mat in range(num_mats):
        print("Material Name: ", problem.mats.get_name(mat))
        print("Scattering Matrix")
        print(problem.mats.get_sigs(mat))
        for group in range(num_groups):
            print("Group Number ", group)
            print("Total XS: ", problem.mats.get_sigt(mat, group))
            print("Fission XS: ", problem.mats.get_sigf(mat, group))
            print("Absorption XS: ", problem.mats.get_siga(mat, group))
            print("Nu: ", problem.mats.get_nu(mat, group))

@filename_to_problem
def profiling(problem):
    source = 10*np.ones(problem.n_elements)
    cProfile.run('problem.op.solve_outer(10*np.ones(problem.n_elements))')

@filename_to_problem
def test_multigroup(problem):
    H = problem.op.build_scattering_matrix()
    source = np.ones(problem.n_elements)
    q = problem.op.make_external_source(source)
    phi = np.zeros(problem.n_nodes)
    print(problem.op.gauss_seidel(H[0, 0], q, phi, tol=1e-5))

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

@filename_to_problem
def plot_from_file(problem):
    phis = np.zeros((problem.num_groups, problem.n_nodes))
    angs = np.zeros((problem.num_groups, 4, problem.n_nodes))
    for g in range(problem.num_groups):
        phis[g] = np.loadtxt("scalar_flux" + str(g))
        for i in range(4):
            angs[g, i] = np.loadtxt("angular_flux_ang" + str(i) + "_group" + str(g))

    for g in range(problem.num_groups):
        scalar_flux = phis[g]
        plot(problem.grid, scalar_flux, problem.filename + "_scalar_flux" + "_group" + str(g))
        ang_fluxes = angs[g]
        for i in range(4):
            plot(problem.grid, ang_fluxes[i], problem.filename + "_ang" + str(i) + "_group" + str(g))


#plot1d(problem.filename)
#test_1d(problem.filename)
#make_lhs(problem.filename)
test_problem("symmetric_fine", "scattering1g", "saaf")
#plot_from_file("std", "fission", "stdfission")
#get_mat_stats("3A", "3A", "3A")
#test_multigroup("std", "scattering1g", "test")
#profiling("symmetric_fine", "scattering1g", "test")
#plot_mats(problem.filename)
