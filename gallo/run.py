import numpy as np
import matplotlib.pyplot as plt

from gallo.formulations.nda import NDA
from gallo.formulations.saaf import SAAF
from gallo.formulations.diffusion import Diffusion
from gallo.fe import FEGrid
from gallo.materials import Materials
from gallo.plot import plot
from gallo.solvers import Solver


def run(nodefile, elefile, matfile, eq_type, eigenvalue, source, name):
    # Setup Problem
    grid = FEGrid(nodefile, elefile)
    mats = Materials(matfile)
    if eq_type == 'NDA':
        op = NDA(grid, mats)
        ua_bool = False
    elif eq_type == 'TGNDA':
        op = NDA(grid, mats)
        ua_bool = True
    elif eq_type == 'SAAF':
        op = SAAF(grid, mats)
    elif eq_type == 'Diffusion':
        op = Diffusion(grid, mats)
    else:
        raise Exception("Equation type not supported.")
    solver = Solver(op)

    if eq_type == 'NDA' or eq_type == 'TGNDA':
        fluxes = solver.solve(source, ua_bool=ua_bool, eigenvalue=eigenvalue)
    else:
        fluxes = solver.solve(source, eigenvalue=eigenvalue)

    # Print Eigenvalue
    if eigenvalue:
        print("Eigenvalue: ", fluxes['k'])

    # Save Fluxes
    phis = fluxes['Phi']
    np.savetxt(eq_type + "_" + name + ".out", phis)

    # Plot Everything
    for g in range(mats.get_num_groups()):
        scalar_flux = phis[g]
        plot(grid, scalar_flux, name + "_scalar_flux" + "_group" + str(g))
