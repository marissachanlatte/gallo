from nose.tools import *
import numpy as np
from numpy.testing import *
import sys
sys.path.append('../src')

from materials import Materials


class TestMaterials:
  # Tests to verify the materials class is working

    @classmethod
    def setup_class(cls):
        cls.filename = "test/test_inputs/test.mat"
        cls.materials = Materials(cls.filename)
        cls.multigroup = "test/test_inputs/multigroup_test.mat"

    def test_read(self):
        eq_(self.materials.num_mats, 2, "number of materials")
        eq_(self.materials.get_name(0), "'core'", "material name")
        eq_(self.materials.get_name(1), "'reflector'", "material name")
        eq_(self.materials.get_siga(0, 0), 5, "siga")
        eq_(self.materials.get_siga(0, 1), 3, "siga")
        eq_(self.materials.get_siga(1, 0), 3, "siga")
        eq_(self.materials.get_siga(1, 1), 2, "siga")
        eq_(self.materials.get_sigf(0, 0), 3, "sigf")
        eq_(self.materials.get_sigf(0, 1), 1, "sigf")
        eq_(self.materials.get_sigf(1, 0), 0, "sigf")
        eq_(self.materials.get_sigf(1, 1), 0, "sigf")
        eq_(self.materials.get_nu(0, 0), 2.43, "nu")
        eq_(self.materials.get_nu(0, 1), 1.43, "nu")
        eq_(self.materials.get_nu(1, 0), 0, "nu")
        eq_(self.materials.get_nu(1, 1), 0, "nu")

        # Test scattering
        mats1 = self.materials.get_sigs(0)
        mats2 = np.array([[1, 0],
                          [0, 2]])
        assert_array_equal(mats1, mats2)
        mats1 = self.materials.get_sigs(1)
        mats2 = np.array([[3, 0],
                          [0, 1]])
        assert_array_equal(mats1, mats2)
