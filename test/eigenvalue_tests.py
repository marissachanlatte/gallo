import sys
sys.path.append('../src')

from nose.tools import *
import numpy as np
from numpy.testing import *
from numpy import linalg

import eigenvalue

class TestEig:
    def test_power_iteration(self):
        A = np.diag((1, 2, 3))
        v_init = np.ones(3)
        eig_numpy, vec_numpy = np.linalg.eig(A)
        eig_gallo, vec_gallo = eigenvalue.power_iteration(A, v_init, tol=1e-8)
        dominant_eig = eig_numpy.max()
        assert_almost_equals(eig_gallo, dominant_eig, places=4)
        print("dominant_eig: ", eig_numpy.max())
        assert_array_almost_equal(vec_gallo,
                                   vec_numpy[np.argmax(eig_numpy)], decimal=4)
