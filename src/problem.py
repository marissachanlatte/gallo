from typing import NamedTuple

from formulations.diffusion import Diffusion
from fe import FEGrid
from materials import Materials
from solvers import Solver


class Problem(NamedTuple):
    op: Diffusion
    grid: FEGrid
    mats: Materials
    solver: Solver
    filename: str

    @property
    def n_elements(self):
        return self.grid.num_elts

    @property
    def num_groups(self):
        return self.mats.get_num_groups()

    @property
    def matrix(self):
        return self.op.get_matrix("all")

    @property
    def n_nodes(self):
        return self.grid.get_num_nodes()
