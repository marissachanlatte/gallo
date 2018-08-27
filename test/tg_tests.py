from nose.tools import *
from numpy.testing import *
import numpy as np

from gallo.formulations.nda import NDA
from gallo.formulations.saaf import SAAF
from gallo.upscatter_acceleration import UA
from gallo.fe import FEGrid
from gallo.materials import Materials

class TestNDA:
    @classmethod
    def setup_class(cls):
        cls.nodefile = "test/test_inputs/symmetric.node"
        cls.elefile = "test/test_inputs/symmetric.ele"
        cls.matfile = "test/test_inputs/3gtest.mat"
        cls.fegrid = FEGrid(cls.nodefile, cls.elefile)
        cls.mats = Materials(cls.matfile)
        cls.op = NDA(cls.fegrid, cls.mats)
        cls.num_groups = cls.mats.get_num_groups
        cls.upscatter_accelerator = UA(cls.op)

    def eigenfunction_test(self):
        eigenfunction = self.upscatter_accelerator.compute_eigenfunction(0)
        assert_array_almost_equal(eigenfunction, np.array([0.4, 0.4, 0.2]), 1e-15)
