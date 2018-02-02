import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
from itertools import *
import sys
sys.path.append('../../src')

from formulations.saaf import SAAF
from fe import *
from materials import Materials
from problem import Problem
from plot import plot

def to_problem(filename):
    nodefile = "../test_inputs/" + filename + ".node"
    elefile = "../test_inputs/" + filename + ".ele"
    matfile = "../test_inputs/" + filename + ".mat"
    grid = FEGrid(nodefile, elefile)
    mats = Materials(matfile)
    op = SAAF(grid, mats)
    A = op.get_matrix("all")
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
    num_nodes = problem.n_nodes
    angular_flux = np.zeros(num_nodes)
    for n in range(num_nodes):
        node = problem.grid.node(n).get_position()
        x = node[0]
        y = node[1]
        xbar = (1 - np.sin(mu))/2 + np.sin(mu)*x
        ybar = (1 - np.sin(eta))/2 + np.sin(eta)*y
        if ybar < np.abs(eta/mu)*xbar:
            angular_flux[n] = inv_sigt*(1 - np.exp((-sigt*ybar)/np.abs(eta)))
        elif ybar > np.abs(eta/mu)*xbar:
            angular_flux[n] = inv_sigt*(1 - np.exp((-sigt*xbar)/np.abs(mu)))
        else:
            print(xbar, ybar)
            raise Exception("ybar=xbar")
    return angular_flux

@filename_to_problem
def fixed_source_error(problem):
    source_terms = np.zeros(problem.n_elements)
    for i in range(problem.n_elements):
        cent = problem.grid.centroid(i)
        # Source Everywhere
        source_terms[i] = 1
    internal_nodes = []
    # Four calculations, S4
    errors = np.zeros(4)
    # Solve for angular flux
    ang_one = 0.3500212
    ang_two = 0.8688903
    angles = product([ang_one, ang_two], repeat=2)
    source = np.ones(problem.n_elements)
    for i, ang in enumerate(angles):
        angle_prod = np.inner(ang, ang)
        lhs = problem.op.make_lhs(angle_prod)[0]
        ang_flux = problem.op.solve(lhs, source, "fixed_source", 0, ang)
        # Calculate MMS solution
        mms_flux = mms_solution(problem, ang)
        # Error
        errors[i] = np.max(np.abs(ang_flux - mms_flux))
    avg_error = np.mean(errors)
    return avg_error

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
        print(np.sqrt(area), " ", norm[inp])
    mms_plot(areas, norm, "mms_saaf")


mms_convergence_test()












