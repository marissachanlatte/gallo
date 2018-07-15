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
    n_elements = grid.get_num_elts()
    num_groups = mats.get_num_groups()
    op = NDA(grid, mats)
    solver = Solver(op)

    return Problem(op=op, mats=mats, grid=grid, solver=solver, filename=filename)

def filename_to_problem(func):
    def _filename_to_problem(mesh, mat, filename):
        return func(problem=to_problem(mesh, mat, filename))
    return _filename_to_problem

@filename_to_problem
def test_problem(problem):
    # source = np.zeros((problem.num_groups, problem.n_elements))
    # for e in range(problem.n_elements):
    #     centroid = problem.grid.centroid(e)
    #     #if .25 < centroid[0] < .75 and .25 < centroid[1] < .75:
    #     if centroid[0] < .5:
    #         source[0, e] = .75
    #         source[1, e] = .25
    ### BOX SOURCE ###
    source = np.zeros((problem.num_groups, problem.n_elements))
    for e in range(problem.n_elements):
        centroid = problem.grid.centroid(e)
        if np.abs(centroid[0]) < 10 and np.abs(centroid[1]) < 10:
            source[0, e] = 7.39
            source[1, e] = 2.61
    # source = np.ones((problem.num_groups, problem.n_elements))
    phis = problem.solver.solve(source, ua_bool=False)
    # Plot Everything
    for g in range(problem.num_groups):
        scalar_flux = phis[g]
        plot(problem.grid, scalar_flux, problem.filename + "_scalar_flux" + "_group" + str(g))

@filename_to_problem
def test_1d(problem):
    source = np.ones((problem.num_groups, problem.n_elements))
    scalar_flux= problem.solver.solve(source, ua_bool=False)
    for g in range(problem.num_groups):
        plot1d(scalar_flux[g], problem.filename + str(g) + "_scalar_flux_1d", -0.875)

def plot1d(sol, filename, y):
    # for symmetric_fine mesh
    # if y==0.125:
    #     nodes = np.array([0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1])
    #     flux = np.array([sol[72], sol[33], sol[62], sol[32], sol[48], sol[28], sol[56], sol[36], sol[77]])
    #     i = 2
    # if y==0.0625:
    #     nodes = np.array([0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375])
    #     flux = np.array([sol[118], sol[117], sol[138], sol[89], sol[137], sol[112], sol[96], sol[124]])
    #     i = 1
    # if y==0:
    #     nodes = np.array([0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1])
    #     flux = np.array([sol[0], sol[71], sol[17], sol[49], sol[8], sol[69], sol[24], sol[78], sol[1]])
    #     i=0
    # for origin_centered10_fine mesh
    if y == 0:
        nodes = np.array([-1,    -0.875,   -0.75,   -0.625,   -0.5,    -0.375,   -0.25,   -0.125,    0,      0.125,    0.25,    0.375,    0.5,     0.625,    0.75,    0.875,    1])
        flux = np.array([sol[5], sol[251], sol[45], sol[240], sol[13], sol[260], sol[58], sol[261], sol[4], sol[238], sol[80], sol[204], sol[15], sol[243], sol[47], sol[257], sol[7]])
    if y == -0.875:
        nodes = np.array([-1,      -0.875,   -0.75,    -0.625,   -0.5,     -0.375,   -0.25,    -.125,   0,        0.125,    0.25,     0.375,    0.5,      0.625,   0.75,     0.875,    1])
        flux = np.array([sol[273], sol[118], sol[193], sol[117], sol[227], sol[138], sol[153], sol[89], sol[248], sol[135], sol[234], sol[112], sol[176], sol[96], sol[200], sol[124], sol[281]])
    plt.plot(nodes, flux, marker="o")
    plt.title("Scalar Flux Line Out at y=" + str(y))
    plt.savefig(filename)
    plt.clf()
    plt.close()

@filename_to_problem
def make_lhs(problem):
    source = np.ones((problem.num_groups, problem.n_elements))
    A = problem.op.make_lhs(0, ho_sols=0)
    print(A)

@filename_to_problem
def test_triangulation(problem):
    triang = problem.grid.setup_triangulation()
    plt.figure()
    plt.triplot(triang)
    plt.savefig(problem.filename)

@filename_to_problem
def test_phi_at_gnodes(problem):
    triang = problem.grid.setup_triangulation()
    phi_prev = np.ones((problem.num_groups, problem.grid.num_nodes))
    for e in range(problem.grid.num_elts):
        g_nodes = problem.grid.gauss_nodes(e)
        phi_vals = problem.grid.phi_at_gauss_nodes(triang, phi_prev, g_nodes)
        print("Element:", e)
        print(g_nodes)

#test_phi_at_gnodes("iron-water-coarse4", "iron-water", "nda_triangles")
#make_lhs("std3", "std3", "test")
test_problem("iron-water-fine", "iron-water", "nda_iron-water")
#test_1d("origin_centered10_fine", "scattering2g", "1d_test")
