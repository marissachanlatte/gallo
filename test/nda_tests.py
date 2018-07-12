from nose.tools import *
from numpy.testing import *
import numpy as np
import sys
sys.path.append('../src')

from formulations.nda import *
from formulations.saaf import *
from fe import *
from materials import *

class TestNDA:
    @classmethod
    def setup_class(cls):
        cls.nodefile = "test/test_inputs/std3.node"
        cls.elefile = "test/test_inputs/std3.ele"
        cls.matfile = "test/test_inputs/noscatter.mat"
        cls.fegrid = FEGrid(cls.nodefile, cls.elefile)
        cls.mats = Materials(cls.matfile)
        cls.op = NDA(cls.fegrid, cls.mats)

    def symmetry_test(self):
        num_groups = self.mats.get_num_groups()
        num_nodes = self.fegrid.get_num_nodes()
        source = np.ones(self.fegrid.get_num_elts())
        ho_phis = np.ones(num_nodes)
        ho_psis = np.ones((4, num_nodes))
        ho_sols = [ho_phis, ho_psis]
        A = self.op.make_lhs(0, ho_sols)
        diff = A.A - A.transpose().A
        ok_(np.allclose(A.A, A.transpose().A, rtol=1e-6))

    def kappa_test(self):
        normal = np.array([0, 1])
        psi = np.ones((4, 2))
        phi = 4*np.pi*psi[0]
        kappa_prime = np.array([.5773503, .5773503])
        kappa = self.op.compute_kappa(normal, phi, psi)
        assert_array_equal(kappa - kappa_prime, np.zeros(2))

    def drift_test(self):
        inv_sigt = 1
        D = 1/3
        ngrad = np.array([1, 1])
        psi = np.ones((4, 2))
        phi = 4*np.pi*psi[0]
        drift_vector = compute_drift_vector(inv_sigt, D, ngrad, phi, psi, nid)
