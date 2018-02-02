from nose.tools import *
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
import sys
sys.path.append('../src')

from formulations.saaf import SAAF
from fe import *
from materials import Materials
from plot import plot

class TestDiffusion():
    @classmethod
    def setup_class(cls):
        cls.nodefile = "test/test_inputs/box_source.node"
        cls.elefile = "test/test_inputs/box_source.ele"
        cls.matfile = "test/test_inputs/box_source.mat"
        cls.fegrid = FEGrid(cls.nodefile, cls.elefile)
        cls.materials = Materials(cls.matfile)
        cls.operator = SAAF(cls.fegrid, cls.materials)

    def test_matrix(self):
        A = self.operator.get_matrix(0)
        assert (A!=A.transpose()).nnz==0
        assert (A.diagonal() >= 0).all()