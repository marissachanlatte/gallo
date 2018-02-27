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

def mms_solution(problem, angles):
    mu = angles[0]
    eta = angles[1]
    inv_sigt = problem.mats.get_inv_sigt(0, 0)
    sigt = problem.mats.get_sigt(0, 0)
    num_nodes = problem.grid.get_num_nodes()
    angular_flux = np.zeros(num_nodes)
    for n in range(num_nodes):
        node = problem.grid.node(n)
        x, y = node.get_position()
        xbar = (1 - np.sin(mu))/2 + np.sin(mu)*x
        ybar = (1 - np.sin(eta))/2 + np.sin(eta)*y
        if ybar <= np.abs(eta/mu)*xbar:
            angular_flux[n] = inv_sigt*(1 - np.exp((-sigt*ybar)/np.abs(eta)))
        elif ybar > np.abs(eta/mu)*xbar:
            angular_flux[n] = inv_sigt*(1 - np.exp((-sigt*xbar)/np.abs(mu)))
    return angular_flux

@filename_to_problem
def fixed_source_error(problem):
    np.set_printoptions(5, suppress=True)
    # source_terms = np.zeros(problem.n_elements)
    # for i in range(problem.n_elements):
    #     cent = problem.grid.centroid(i)
    #     # Source Everywhere
    #     source_terms[i] = 1/problem.n_elements
    ang_one = .5773503
    ang_two = -.5773503
    angles = product([ang_one, ang_two], repeat=2)
    source = np.ones(problem.n_elements)
    #mms_scalar = 0
    nodes = problem.grid.get_num_nodes()
    phi_prev = np.zeros(nodes)
    ang = [ang_one, ang_one]
    A = problem.op.make_rhs(0, source, ang, "vacuum", phi_prev)
    #print(A)
    ### CALCULATE & PLOT MMS SOLUTION ###
    # for i, ang in enumerate(angles):
    #     mms_flux = mms_solution(problem, ang)
    #     mms_scalar += mms_flux
    #     plot(problem.grid, mms_flux, "mms" + str(i))
    # mms_scalar /= 4
    # plot(problem.grid, mms_scalar, "mms_scalar")

    # scalar_flux, ang_fluxes = problem.op.solve(source, "eigenvalue", 0, "reflecting", tol=1e-1)
    # for i, ang in enumerate(angles):
    #     plot(problem.grid, ang_fluxes[i], "saaf" + str(i))
    # plot(problem.grid, scalar_flux, "scalar_flux", mesh_plot=True)
    
@filename_to_problem
def test_problem(problem):
    source = np.ones(problem.n_elements)
    scalar_flux, ang_fluxes = problem.op.solve(source, "eigenvalue", 0, "vacuum", tol=1e-1)
    for i in range(4):
        plot(problem.grid, ang_fluxes[i], "saaf" + str(i))
    plot(problem.grid, scalar_flux, "scalar_flux")

@filename_to_problem
def check_symmetry(problem):
    source = np.ones(problem.n_elements)
    ang_one = .5773503
    ang_two = .5773503
    A = problem.op.make_lhs(np.array([ang_one, ang_two]))[0]
    nonzero = (A!=A.transpose()).nonzero()
    #print(A[6, 60] - A[60, 6])
    #print((A!=A.transpose()).nnz==0)
    print(np.allclose(A.A, A.transpose().A, rtol=1e-12))
def oned_solution():
    x = np.linspace(0, 1, 100)
    phi = (1 - ((np.exp(np.sqrt(6)*(1-x)) + np.exp(np.sqrt(6)*x))
        /(1 - 1/np.sqrt(2) + (1 + 1/np.sqrt(2))*np.exp(6))))
    plt.plot(x, phi)
    plt.savefig("1d_solution")

def twod_solution():
    num_nodes = problem.grid.get_num_interior_nodes()
    phi = np.zeros(num_nodes)
    for n in range(num_nodes):
        node = problem.grid.interior_node(n).get_position()
        x = node[0]
        y = node[1]
        xbar = .5 + np.sqrt((x-.5)**2 + (y-.5)**2)
        phi[n] = (1 - ((np.exp(np.sqrt(6)*(1-xbar)) + np.exp(np.sqrt(6)*xbar))
        /(1 - 1/np.sqrt(2) + (1 + 1/np.sqrt(2))*np.exp(6))))
    phi = reinsert(problem.grid, phi)
    plot(problem.grid, phi, "2d_solution")

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

def mms_convergence_test():
    N = 3
    areas = np.zeros(N)
    norm = np.zeros(N)
    for inp in range(N):
        problem = to_problem("mesh" + str(inp))
        # Compute average error
        norm[inp] = fixed_source_error(problem.filename)
        area = problem.grid.average_element_area()
        areas[inp] = area
        #print(np.sqrt(area), " ", norm[inp])
    mms_plot(areas, norm, "mms_saaf")


#mms_convergence_test()
problem = to_problem("saaf")
print("Scattering XS: ", problem.mats.get_sigs(0, 0))
print("Total XS: ", problem.mats.get_sigt(0, 0))
#fixed_source_error(problem.filename)
test_problem(problem.filename)
#mesh_plot(problem.grid, problem.filename)
#check_symmetry(problem.filename)












