import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np


def _setup_triangles(grid):
    # Get number of nodes
    nodes = grid.num_nodes
    # Setup xy
    x = np.zeros(nodes)
    y = np.zeros(nodes)
    positions = (grid.node(i).position for i in range(nodes))
    for i, pos in enumerate(positions):
        x[i], y[i] = pos
    # Setup triangles
    elts = grid.num_elts
    triangles = np.array([grid.element(i).vertices for i in range(elts)])
    triang = tri.Triangulation(x, y, triangles=triangles)
    return triang

def plot(grid, solution, filename=None, savefig=True):
    triang = _setup_triangles(grid)
    # Interpolate to Refined Triangular Grid
    interp_lin = tri.LinearTriInterpolator(triang, solution)
    refiner = tri.UniformTriRefiner(triang)
    tri_refi, sol_refi = refiner.refine_field(solution, triinterpolator=interp_lin, subdiv=3)
    # Setup colorbar
    inf = np.min(sol_refi)
    sup = np.max(sol_refi)
    # Plot and save to file
    plt.figure()
    plt.triplot(triang)
    if inf != sup:
        plt.tricontourf(tri_refi, sol_refi, levels=np.linspace(inf, sup, 11))
    else:
        plt.tricontourf(tri_refi, sol_refi)
    plt.colorbar()
    if savefig:
        plt.savefig(filename)
        plt.clf()
        plt.close()

def plot_mesh(grid, filename):
    triang = _setup_triangles(grid)
    elts = grid.num_elts
    mats = np.zeros(elts)
    for i in range(elts):
        el = grid.element(i)
        mats[i] = el.mat_id
    plt.figure()
    plt.triplot(triang)
    plt.tripcolor(triang, mats, shading='flat')
    plt.colorbar()
    plt.savefig(filename)
    plt.clf()
    plt.close()

def plot_1d(grid, solution, y_fixed, save=False, filename=None):
    fluxes = []
    for node in range(grid.num_nodes):
        x, y = grid.node(node).position
        if y == y_fixed:
            fluxes.append([x, solution[node]])
    # Sort Fluxes by X Val
    fluxes = np.array(fluxes)
    fluxes = fluxes[fluxes[:, 0].argsort()]
    return fluxes
