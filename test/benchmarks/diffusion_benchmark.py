import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import sys
sys.path.append('../../src')

from formulations.diffusion import Diffusion
from fe import *
from materials import Materials
from problem import Problem
from plot import plot
from solvers import *

def to_problem(mesh, mat, filename):
    nodefile = "../test_inputs/" + mesh + ".node"
    elefile = "../test_inputs/" + mesh + ".ele"
    matfile = "../test_inputs/" + mat + ".mat"
    grid = FEGrid(nodefile, elefile)
    mats = Materials(matfile)
    op = Diffusion(grid, mats)
    #A = op.get_matrix("all")
    solver = Solver(op)
    n_elements = grid.get_num_elts()
    num_groups = mats.get_num_groups()
    return Problem(op=op, mats=mats, grid=grid, solver=solver, filename=filename)

def filename_to_problem(func):
    def _filename_to_problem(mesh, mat, filename):
        return func(problem=to_problem(mesh, mat, filename))
    return _filename_to_problem

def source_function(x, problem, filename):
    if filename=="uniform_source":
        return 1

    elif filename=="box_source":
        if 4 < x[0] < 6 and 4 < x[1] < 6:
            return 1
        else:
            return 0

    elif "mesh" in filename:
        D = problem.mats.get_diff(0, 0)
        mms = np.sin(x[0]*np.pi)*np.sin(x[1]*np.pi)
        return 2*D*np.pi**2*mms + mms

    else:
      print("Input not supported")

@filename_to_problem
def fixed_source_test(problem):
    source_terms = np.zeros(problem.n_elements)
    for i in range(problem.n_elements):
      cent = problem.grid.centroid(i)
      source_terms[i] = source_function(cent, problem, problem.filename)
    internal_nodes = []
    for g in range(problem.num_groups):
        rhs = problem.op.make_rhs(source_terms)
        internal_nodes.append(problem.op.solve(problem.matrix[g], rhs, "fixed_source", g))
    return internal_nodes, problem.grid, problem.mats

@filename_to_problem
def eigenvalue_test(problem):
    for g in range(problem.num_groups):
        phi, k = problem.op.solve(problem.matrix[g], None, "eigenvalue", g, 1000, 1e-5)
    k_exact = (problem.mats.get_nu(0, 0)*problem.mats.get_sigf(0, 0)/
        (problem.mats.get_siga(0, 0) + 2*problem.mats.get_diff(0, 0)*np.pi**2))
    err = np.abs(k_exact - k)
    area = problem.grid.average_element_area()
    flux = reinsert(problem.grid, phi)
    plot(problem.grid, flux, "diffusion_eigenvalue" + str(problem.filename))
    print("K-Eigenvalue: ", k)
    print("Error: ", err)
    return err, area, phi, k

def phi_plot(grid, internal_nodes, filename, group_id):
    phi = reinsert(grid, internal_nodes)
    plot(grid, phi, filename + "_group" + str(group_id) + "_test")

def mms_plot(area, err, plotname):
    plt.close()
    area = np.sqrt(area)
    fit = np.polyfit(np.log(area), np.log(err), 1)
    print("Slope: ", fit[0])
    f = lambda x: np.exp(fit[1]) * x**(fit[0])
    plt.xlabel("Square Root of Average Element Area")
    plt.ylabel("Max Error")
    plt.loglog(area, err, "-o")
    plt.loglog(area, f(area))
    plt.savefig(plotname + "_plot")

def kmms():
    N = 3
    err = np.zeros(N)
    area = np.zeros(N)
    for inp in range(N):
        filename = "eigenvalue" + str(inp)
        err[inp], area[inp], *_ = eigenvalue_test(filename)
    mms_plot(area, err, "kmms")

def mms():
    N = 3
    areas = np.zeros(N)
    norm = np.zeros(N)
    for inp in range(N):
        problem = to_problem("mesh" + str(inp))
        internal_nodes, *_ = fixed_source_test(problem.filename)
        # Compute Exact Solution
        inodes = problem.grid.get_num_interior_nodes()
        exact = np.zeros(inodes)
        err = np.zeros(inodes)
        for i in range(problem.n_elements):
            for j in range(3):
                node = problem.grid.get_node(i, j)
                if node.is_interior():
                    ID = node.get_interior_node_id()
                    x = node.get_position()
                    exact[ID] = np.sin(x[0]*np.pi)*np.sin(x[1]*np.pi)
        # Calculate norm
        norm[inp] = np.max(np.abs(internal_nodes[0] - exact)) #inf norm
        area = problem.grid.average_element_area()
        areas[inp] = area
        print(np.sqrt(area), " ", norm[inp])
    mms_plot(areas, norm, "mms")

@filename_to_problem
def test_problem(problem):
    source = np.ones((problem.num_groups, problem.n_elements))
    #phis, angs, eigenvalue = problem.op.solve(source, eigenvalue=True)
    phis = problem.solver.solve(source, eigenvalue=False)

    # Plot Everything
    for g in range(problem.num_groups):
        scalar_flux = phis[g]
        plot(problem.grid, scalar_flux, problem.filename + "_scalar_flux" + "_group" + str(g))

test_problem("origin_centered10_fine", "scattering2g", "test")
