from nose.tools import *
import numpy as np
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
        cls.multi_mats = Materials(cls.multigroup)

    def test_read(self):
        eq_(self.materials.num_mats, 2, "number of materials")
        eq_(self.materials.get_name(0), "'core'", "material name")
        eq_(self.materials.get_name(1), "'reflector'", "material name")
        eq_(self.materials.get_siga(0, 0), 5, "siga")
        eq_(self.materials.get_siga(0, 1), 3, "siga")
        eq_(self.materials.get_siga(1, 0), 3, "siga")
        eq_(self.materials.get_siga(1, 1), 2, "siga")
        eq_(self.materials.get_sigs(0, 0), 1, "sigs")
        eq_(self.materials.get_sigs(0, 1), 2, "sigs")
        eq_(self.materials.get_sigs(1, 0), 3, "sigs")
        eq_(self.materials.get_sigs(1, 1), 1, "sigs")
        eq_(self.materials.get_sigf(0, 0), 3, "sigf")
        eq_(self.materials.get_sigf(0, 1), 1, "sigf")
        eq_(self.materials.get_sigf(1, 0), 0, "sigf")
        eq_(self.materials.get_sigf(1, 1), 0, "sigf")
        eq_(self.materials.get_nu(0, 0), 2.43, "nu")
        eq_(self.materials.get_nu(0, 1), 1.43, "nu")
        eq_(self.materials.get_nu(1, 0), 0, "nu")
        eq_(self.materials.get_nu(1, 1), 0, "nu")

    def test_multigroup(self):
        eq_(self.multi_mats.get_sigs(0, 0), .01)
        eq_(self.multi_mats.get_sigs(0, 1), .02)



