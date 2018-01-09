import sys
sys.path.append('../src')

from plot import plot
from fe import *

nodefile = "test_inputs/test.node"
elefile = "test_inputs/test.ele"
fegrid = FEGrid(nodefile, elefile)

nodes = fegrid.get_num_nodes()
nnodes = range(nodes)
solution = np.zeros(nodes)
positions = (fegrid.node(i).get_position() for i in nnodes)
for i, pos in enumerate(positions):
    if pos[0] >= .5 and pos[1] >= .5:
        solution[i] = 1

plot(fegrid, solution, "test_output", mesh_plot=True)