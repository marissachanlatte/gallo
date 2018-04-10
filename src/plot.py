import matplotlib.pyplot as plt 
import matplotlib.tri as tri
import numpy as np 


def plot(grid, solution, filename, mesh_plot=False):
    # Get number of nodes
    nodes = grid.get_num_nodes()
    # Setup xy
    x = np.zeros(nodes)
    y = np.zeros(nodes)
    positions = (grid.node(i).get_position() for i in range(nodes))
    for i, pos in enumerate(positions):
        x[i], y[i] = pos
    # Setup triangles
    elts = grid.get_num_elts()
    triangles = np.array([grid.element(i).get_vertices() for i in range(elts)])
    triang = tri.Triangulation(x, y, triangles=triangles)
    # Interpolate to Refined Triangular Grid
    interp_lin = tri.CubicTriInterpolator(triang, solution)
    refiner = tri.UniformTriRefiner(triang)
    tri_refi, sol_refi = refiner.refine_field(solution, triinterpolator=interp_lin, subdiv=3)
    # Setup colorbar
    inf = np.min(sol_refi)
    sup = np.max(sol_refi)
    # Plot and save to file
    plt.figure()
    plt.triplot(triang)
    plt.tricontourf(tri_refi, sol_refi, levels=np.linspace(inf, sup, 11))
    plt.colorbar()
    plt.savefig(filename)
    plt.clf()
    plt.close()

