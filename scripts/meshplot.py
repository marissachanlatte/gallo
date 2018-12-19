from gallo.plot import plot_mesh
from gallo.fe import FEGrid

file = "3A"
filepath = "../test/test_inputs/" + file
grid = FEGrid(filepath + ".node", filepath + ".ele")
filename = file + "_mesh"

plot_mesh(grid, filename)
