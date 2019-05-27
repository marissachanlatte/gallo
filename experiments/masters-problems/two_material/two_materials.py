import numpy as np

from gallo.run import run
from gallo.fe import *

# Run File for Two Material Problem

# Path to input files
nodefile = "inputs/iron-water10.node"
elefile = "inputs/iron-water10.ele"
matfile = "inputs/mod-water.mat"

# Equation Type: NDA, TGNDA, SAAF, or Diffusion
eq_type = "NDA"
eigenvalue = False

# Setup Source. Must be an array of size (num groups, num elements)
grid = FEGrid(nodefile, elefile)
source = np.zeros((7, 256))
for g in range(3):
    for e in range(256):
        centroid = grid.centroid(e)
        if np.abs(centroid[0]) < 2.5 and np.abs(centroid[1]) < 2.5:
            if g == 0:
                source[g, e] = 7
            elif g == 1:
                source[g, e] = 2
            elif g == 2:
                source[g, e] = 1
# Problem Name
name = 'nda_two_material'

run(nodefile, elefile, matfile, eq_type, eigenvalue, source, name)
