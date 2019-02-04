import numpy as np

from gallo.run import run

# Example Run File for One Group, No Scattering Problem

# Path to input files
nodefile = "inputs/symmetric_fine.node"
elefile = "inputs/symmetric_fine.ele"
matfile = "inputs/noscatter.mat"

# Equation Type: NDA, TGNDA, SAAF, or Diffusion
eq_type = "NDA"
eigenvalue = False

# Setup Source. Must be an array of size (num groups, num elements)
source = np.ones((1, 256))

# Problem Name
name = 'no-scattering'

run(nodefile, elefile, matfile, eq_type, eigenvalue, source, name)
