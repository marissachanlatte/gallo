import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import sys
sys.path.append('../src')

from diffusion import Diffusion
from fe import *
from materials import Materials
from plot import plot

def source_function(x, filename):
    if filename=="uniform_source":
        return 1

    elif filename=="box_source":
        if 4 < x[0] < 6 and 4 < x[1] < 6:
            return 1
        else:
            return 0

    elif filename=="mms":
        D = .16667
        mms = np.sin(x[0]*np.pi)*np.sin(x[1]*np.pi)
        return 2*D*np.pi**2*mms + mms

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

def mms():
    N = 3
    areas = np.zeros(N)
    norm = np.zeros(N)
    for inp in range(N):
        filename = "mesh" + str(inp)
        # Solve Using Gallo Code
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
          source_terms[i] = source_function(cent, "mms")
        rhs = op.make_rhs(source_terms)
        internal_nodes = linalg.cg(A, rhs)
        phi = reinsert(grid, internal_nodes[0])
        # Compute Exact Solution
        inodes = grid.get_num_interior_nodes()
        exact = np.zeros(inodes)
        err = np.zeros(inodes)
        for i in range(n_elements):
            for j in range(3):
                node = grid.get_node(i, j)
                if node.is_interior():
                    ID = node.get_interior_node_id()
                    x = node.get_position()
                    exact[ID] = np.sin(x[0]*np.pi)*np.sin(x[1]*np.pi)
        # Calculate norm
        for i in range(inodes):
            err[i] = np.abs(internal_nodes[0][i] - exact[i])
        norm[inp] = np.max(err) #inf norm
        #Calculate average area
        area = 0
        for i in range(n_elements):
            area += grid.element_area(i)
        area /= n_elements
        areas[inp] = np.sqrt(area)
        print(np.sqrt(area), " ", np.max(err))
    fit = np.polyfit(np.log(areas), np.log(norm), 1)
    print("Slope: ", fit[0])
    f = lambda x: np.exp(fit[1]) * x**(fit[0])
    plt.xlabel("Square Root of Average Element Area")
    plt.ylabel("Max Error")
    plt.loglog(areas, norm, "-o")
    plt.loglog(areas, f(areas))
    plt.show()
    plt.savefig("mms_plot")


diffusion_test("uniform_source")
diffusion_test("box_source")
mms()
