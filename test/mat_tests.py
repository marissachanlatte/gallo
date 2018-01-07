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

  def test_read(self):
    eq_(self.materials.num_mats, 2, "number of materials")
    eq_(self.materials.get_name(0), "'core'", "material name")
    eq_(self.materials.get_name(1), "'reflector'", "material name")
    eq_(self.materials.get_siga(0), 5, "sigt")
    eq_(self.materials.get_siga(1), 3, "sigt")
    eq_(self.materials.get_sigs(0), 1, "sigt")
    eq_(self.materials.get_sigs(1), 3, "sigt")
    eq_(self.materials.get_sigf(0), 3, "sigt")
    eq_(self.materials.get_sigf(1), 0, "sigt")
    eq_(self.materials.get_nu(0), 2.43, "sigt")
    eq_(self.materials.get_nu(1), 0, "sigt")
