from nose.tools import *
from numpy.testing import *
import numpy as np

from gallo.formulations.nda import NDA
from gallo.formulations.saaf import SAAF
from gallo.fe import FEGrid
from gallo.materials import Materials

class TestNDA:
    @classmethod
    def setup_class(cls):
        cls.nodefile = "test/test_inputs/std3.node"
        cls.elefile = "test/test_inputs/std3.ele"
        cls.matfile = "test/test_inputs/noscatter.mat"
        cls.fegrid = FEGrid(cls.nodefile, cls.elefile)
        cls.mats = Materials(cls.matfile)
        cls.op = NDA(cls.fegrid, cls.mats)

    # def kappa_test(self):
    #     normal = np.array([0, 1])
    #     psi = np.ones((4, 2))
    #     phi = 4*np.pi*psi[0]
    #     kappa_prime = np.array([.5773503, .5773503])
    #     kappa = self.op.compute_kappa(normal, phi, psi)
    #     assert_array_equal(kappa - kappa_prime, np.zeros(2))
    #
    # def drift_test(self):
    #     inv_sigt = 1
    #     D = 1/3
    #     ngrad = np.array([1, 1])
    #     psi = np.ones((4, 3))
    #     phi = 4*np.pi*psi[0]
    #     drift_vector = self.op.compute_drift_vector(inv_sigt, D, ngrad, phi, psi)
    #     ang = np.array([.5773503, .5773503])
    #     d_prime = (2*np.pi*ang*(ang@ngrad) - (4/3*np.pi*ngrad))/phi[0]
    #     ones = np.ones((3, 2))
    #     assert_array_equal(drift_vector, d_prime*ones)
