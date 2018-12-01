from nose.tools import *
from numpy.testing import *
from nose.plugins.attrib import attr
import numpy as np

from gallo.formulations.nda import NDA
from gallo.formulations.saaf import SAAF
from gallo.fe import FEGrid
from gallo.materials import Materials
from gallo.solvers import Solver

class TestNDA:
    @classmethod
    def setup_class(cls):
        cls.nodefile = "test/test_inputs/std3.node"
        cls.elefile = "test/test_inputs/std3.ele"
        cls.matfile = "test/test_inputs/noscatter.mat"
        cls.fegrid = FEGrid(cls.nodefile, cls.elefile)
        cls.mats = Materials(cls.matfile)
        cls.op = NDA(cls.fegrid, cls.mats)
        cls.symnode = "test/test_inputs/symmetric_fine.node"
        cls.symele = "test/test_inputs/symmetric_fine.ele"
        cls.symgrid = FEGrid(cls.symnode, cls.symele)
        cls.nop = NDA(cls.symgrid, cls.mats)
        cls.nosolv = Solver(cls.nop)
        cls.onescatfile = "test/test_inputs/scattering1g.mat"
        cls.onescatmat = Materials(cls.onescatfile)
        cls.oneop = NDA(cls.symgrid, cls.onescatmat)
        cls.onesolv = Solver(cls.oneop)
        cls.orignode = "test/test_inputs/origin_centered10_fine.node"
        cls.origele = "test/test_inputs/origin_centered10_fine.ele"
        cls.origrid = FEGrid(cls.orignode, cls.origele)
        cls.twomatfile = "test/test_inputs/scattering2g.mat"
        cls.twoscatmat = Materials(cls.twomatfile)
        cls.twop = NDA(cls.origrid, cls.twoscatmat)
        cls.twosolv = Solver(cls.twop)

    @attr('slow')
    def test_no_scat(self):
        source = 10*np.ones((self.nop.num_groups, self.nop.num_elts))
        fluxes = self.nosolv.solve(source, eigenvalue=False)
        phis = fluxes['Phi']
        gold_phis = np.array([np.loadtxt("test/test_outputs/nda_no_scat.out")])
        assert_array_almost_equal(phis, gold_phis, decimal=2)

    @attr('slow')
    def test_tg_no_scat(self):
        source = 10*np.ones((self.nop.num_groups, self.nop.num_elts))
        fluxes = self.nosolv.solve(source, ua_bool=True)
        phis = fluxes['Phi']
        gold_phis = np.array([np.loadtxt("test/test_outputs/tgnda_no_scat.out")])
        assert_array_almost_equal(phis, gold_phis, decimal=2)

    @attr('slow')
    def test_tg_1g(self):
        source = 10*np.ones((self.oneop.num_groups, self.oneop.num_elts))
        fluxes = self.onesolv.solve(source, ua_bool=True)
        phis = fluxes['Phi']
        gold_phis = np.array([np.loadtxt("test/test_outputs/tgnda1g.out")])
        assert_array_almost_equal(phis, gold_phis, decimal=2)

    @attr('slow')
    def test_1g(self):
        source = 10*np.ones((self.oneop.num_groups, self.oneop.num_elts))
        fluxes = self.onesolv.solve(source, ua_bool=False)
        phis = fluxes['Phi']
        gold_phis = np.array([np.loadtxt("test/test_outputs/nda1g.out")])
        assert_array_almost_equal(phis, gold_phis, decimal=2)

    @attr('slow')
    def test_2g(self):
        source = np.ones((self.twop.num_groups, self.twop.num_elts))
        fluxes = self.twosolv.solve(source, ua_bool=False)
        phis = fluxes['Phi']
        gold_phis = np.loadtxt("test/test_outputs/nda2g.out")
        assert_array_almost_equal(phis, gold_phis, decimal=2)

    @attr('slow')
    def test_tg_2g(self):
        source = np.ones((self.twop.num_groups, self.twop.num_elts))
        fluxes = self.twosolv.solve(source, ua_bool=True)
        phis = fluxes['Phi']
        gold_phis = np.loadtxt("test/test_outputs/tgnda2g.out")
        assert_array_almost_equal(phis, gold_phis, decimal=2)
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
