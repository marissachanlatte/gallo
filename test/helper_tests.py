from nose.tools import *
from numpy.testing import *
import numpy as np

from gallo.formulations.diffusion import Diffusion
from gallo.fe import FEGrid
from gallo.materials import Materials
from gallo.helpers import Helper

class TestHelpers():
    @classmethod
    def setup_class(cls):
        cls.stdnode = "test/test_inputs/std.node"
        cls.stdele = "test/test_inputs/std.ele"
        cls.stdgrid = FEGrid(cls.stdnode, cls.stdele)
        cls.fissionfile = "test/test_inputs/fissiontest.mat"
        cls.fissionmat = Materials(cls.fissionfile)
        cls.fissop = Diffusion(cls.stdgrid, cls.fissionmat)
        cls.helper = Helper(cls.stdgrid, cls.fissionmat)

    def test_fission_source(self):
        A = self.helper.compute_fission_source(0, np.array([1]), 0)
        assert (A == 1)

    def test_flux_at_elt(self):
        flux = np.random.rand(1, 4)
        flux_at_elt = self.helper.flux_at_elt(flux)
        true_flux = np.array([[(flux[0, 0] + flux[0, 1] + flux[0, 3])/3,
                              (flux[0, 1] + flux[0, 2] + flux[0, 3])/3]])
        assert_array_equal(flux_at_elt, true_flux)

    def test_integrate_flux(self):
        g = 2
        flux = np.random.rand(g, 4)
        integral = self.helper.integrate_flux(flux)
        true_integral = np.array([(flux[i, 0] + flux[i, 1] + flux[i, 3])/3 +
                              (flux[i, 1] + flux[i, 2] + flux[i, 3])/3 for i in range(g)])
        true_integral = np.sum(true_integral)*0.5
        assert_equal(integral, true_integral)
