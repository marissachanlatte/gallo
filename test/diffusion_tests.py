from nose.tools import *
from numpy.testing import *
from nose.plugins.attrib import attr
import numpy as np

from gallo.formulations.diffusion import Diffusion
from gallo.fe import FEGrid
from gallo.materials import Materials
from gallo.plot import plot
from gallo.solvers import Solver

class TestDiffusion():
    @classmethod
    def setup_class(cls):
        cls.stdnode = "test/test_inputs/std.node"
        cls.stdele = "test/test_inputs/std.ele"
        cls.stdgrid = FEGrid(cls.stdnode, cls.stdele)
        cls.fissionfile = "test/test_inputs/fissiontest.mat"
        cls.fissionmat = Materials(cls.fissionfile)
        cls.fissop = Diffusion(cls.stdgrid, cls.fissionmat)
        cls.solver = Solver(cls.fissop)
        cls.symnode = "test/test_inputs/symmetric_fine.node"
        cls.symele = "test/test_inputs/symmetric_fine.ele"
        cls.symgrid = FEGrid(cls.symnode, cls.symele)
        cls.symfissop = Diffusion(cls.symgrid, cls.fissionmat)
        cls.symsolver = Solver(cls.symfissop)
        cls.orignode = "test/test_inputs/origin_centered10_fine.node"
        cls.origele = "test/test_inputs/origin_centered10_fine.ele"
        cls.origrid = FEGrid(cls.orignode, cls.origele)
        cls.twoscatfile = "test/test_inputs/scattering2g.mat"
        cls.twoscatmat = Materials(cls.twoscatfile)
        cls.twop = Diffusion(cls.origrid, cls.twoscatmat)
        cls.twosolv = Solver(cls.twop)
        cls.onescatfile = "test/test_inputs/scattering1g.mat"
        cls.onescatmat = Materials(cls.onescatfile)
        cls.oneop = Diffusion(cls.symgrid, cls.onescatmat)
        cls.onesolv = Solver(cls.oneop)
        cls.noscatfile = "test/test_inputs/noscatter.mat"
        cls.noscatmat = Materials(cls.noscatfile)
        cls.nop = Diffusion(cls.symgrid, cls.noscatmat)
        cls.nosolv = Solver(cls.nop)

    def test_matrix(self):
        A = self.symfissop.make_lhs(0)
        assert (A!=A.transpose()).nnz==0
        assert (A.diagonal() >= 0).all()

    def test_eigenvalue(self):
        source = np.zeros((self.symfissop.num_groups, self.symfissop.num_elts))
        phi, k = self.symsolver.solve(source, eigenvalue=True)
        assert_allclose(k, 0.234582, rtol=0.5)

    @attr('slow')
    def test_two_group(self):
        source = np.ones((self.twop.num_groups, self.twop.num_elts))
        phis = self.twosolv.solve(source, eigenvalue=False)
        gold_phis = np.loadtxt("test/test_outputs/diff2g.out")
        assert_array_almost_equal(phis, gold_phis, decimal=4)

    @attr('slow')
    def test_one_group(self):
        source = np.ones((self.oneop.num_groups, self.oneop.num_elts))
        phis = self.onesolv.solve(source, eigenvalue=False)
        gold_phis = np.array([np.loadtxt("test/test_outputs/diff1g.out")])
        assert_array_almost_equal(phis, gold_phis, decimal=4)

    @attr('slow')
    def test_no_scat(self):
        source = np.ones((self.nop.num_groups, self.nop.num_elts))
        phis = self.nosolv.solve(source, eigenvalue=False)
        gold_phis = np.array([np.loadtxt("test/test_outputs/diff_no_scat.out")])
        assert_array_almost_equal(phis, gold_phis, decimal=4)
