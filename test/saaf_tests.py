from nose.tools import *
from numpy.testing import *
import numpy as np
import sys
sys.path.append('../src')

from formulations.saaf import *
from fe import *
from materials import *

class TestSAAF:
    @classmethod
    def setup_class(cls):
        cls.nodefile = "test/test_inputs/mesh0.node"
        cls.elefile = "test/test_inputs/mesh0.ele"
        cls.matfile = "test/test_inputs/mesh0.mat"
        cls.fegrid = FEGrid(cls.nodefile, cls.elefile)
        cls.mats = Materials(cls.matfile)
        cls.op = SAAF(cls.fegrid, cls.mats)

    def symmetry_test(self):
        source = np.ones(self.fegrid.get_num_elts())
        ang_one = .5773503
        ang_two = -.5773503
        A = self.op.make_lhs(np.array([ang_one, ang_two]))[0]
        nonzero = (A!=A.transpose()).nonzero()
        ok_(np.allclose(A.A, A.transpose().A, rtol=1e-12))
