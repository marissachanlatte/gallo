from nose.tools import *
from numpy.testing import *
import numpy as np
import sys
sys.path.append('../src')

from formulations.saaf import *
from fe import *
from materials import *

class TestSAAF:
    @classmethod
    def setup_class(cls):
        cls.nodefile = "test/test_inputs/mesh0.node"
        cls.elefile = "test/test_inputs/mesh0.ele"
        cls.matfile = "test/test_inputs/mesh0.mat"
        cls.fegrid = FEGrid(cls.nodefile, cls.elefile)
        cls.mats = Materials(cls.matfile)
        cls.op = SAAF(cls.fegrid, cls.mats)

        cls.stdnode = "test/test_inputs/std.1.node"
        cls.stdele = "test/test_inputs/std.1.ele"
        cls.stdmat = "test/test_inputs/std.1.mat"
        cls.stdmats = Materials(cls.matfile)
        cls.stdgrid = FEGrid(cls.stdnode, cls.stdele)
        cls.stdop = SAAF(cls.stdgrid, cls.stdmats)

        cls.smnode = "test/test_inputs/D.1.node"
        cls.smele = "test/test_inputs/D.1.ele"
        cls.smgrid = FEGrid(cls.smnode, cls.smele)
        cls.smop = SAAF(cls.smgrid, cls.mats)


    def symmetry_test(self):
        source = np.ones(self.fegrid.get_num_elts())
        ang_one = .5773503
        ang_two = -.5773503
        A = self.op.make_lhs(np.array([ang_one, ang_two]))[0]
        nonzero = (A!=A.transpose()).nonzero()
        ok_(np.allclose(A.A, A.transpose().A, rtol=1e-12))

    def hand_calculation_test(self):
        angles = np.array([.5773503, .5773503])
        A = self.stdop.make_lhs(angles)[0].todense()
        print(A)
        hand = np.array([[ 0.,          0.,          0.,          0.        ],
                         [ 0.,          0.1924501,   0.09622505,  0.        ],
                         [ 0.,          0.09622505,  0.3849002,   0.09622505],
                         [ 0.,          0.,          0.09622505,  0.1924501 ]])
        ok_(np.allclose(A, hand, rtol=1e-7))

    def incident_angle_test(self):
        ang_one = .5773503
        ang_two = -.5773503
        angles = itr.product([ang_one, ang_two], repeat=2)
        psi_prev = np.array([[0, 1, 2, 3],
                             [4, 5, 6, 7],
                             [8, 9, 10, 11],
                             [12, 13, 14, 15]])
        psi_new = np.zeros((4, 4))
        for i, ang in enumerate(angles):
            for nid in range(4):
                psi_new[i, nid] = self.stdop.assign_incident(nid, ang, psi_prev)
        psi_correct = np.array([[12, 13, 14, 15],
                                [8, 9, 10, 11],
                                [4, 5, 6, 7],
                                [0, 1, 2, 3]])

        assert_array_equal(psi_new, psi_correct)

    def incident_angle_test2(self):
        ang_one = .5773503
        ang_two = -.5773503
        angles = itr.product([ang_one, ang_two], repeat=2)
        psi_prev = np.array([[1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16],
                             [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                             [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48], 
                             [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]])
        psi_new = np.zeros((4, 16))
        for i, ang in enumerate(angles):
            for nid in range(16):
                if not self.smgrid.node(nid).is_interior():
                    psi_new[i, nid] = self.smop.assign_incident(nid, ang, psi_prev)
        psi_correct = np.array([[49, 18, 19, 52, 37, 0, 0, 40, 41, 0, 0, 44, 61, 30, 31, 64], 
                                [33, 2,  3,  36, 53, 0, 0, 56, 57, 0, 0, 60, 45, 14, 15, 48], 
                                [17, 50, 51, 20, 5,  0, 0, 8,  9,  0, 0, 12, 29, 62, 63, 32], 
                                [1,  34, 35, 4,  21, 0, 0, 24, 25, 0, 0, 28, 13, 46, 47, 16]])
        assert_array_equal(psi_new, psi_correct)

                