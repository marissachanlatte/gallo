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
    if mesh_plot:
        # Plot mesh
        plt.figure()
        plt.gca().set_aspect('equal')
        plt.triplot(x, y, triangles, 'go-', lw=1.0)
        plt.savefig(filename + "_mesh")
    # Setup colorbar
    inf = np.min(solution)
    sup = np.max(solution)
    # Plot and save to file
    plt.figure()
    plt.tricontourf(x, y, triangles, solution, levels=np.linspace(inf, sup, 11))
    plt.colorbar()
    plt.savefig(filename)