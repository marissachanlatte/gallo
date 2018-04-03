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

        cls.stdnode = "test/test_inputs/std.node"
        cls.stdele = "test/test_inputs/std.ele"
        cls.stdmatfile = "test/test_inputs/std.mat"
        cls.stdmats = Materials(cls.stdmatfile)
        cls.stdgrid = FEGrid(cls.stdnode, cls.stdele)
        cls.stdop = SAAF(cls.stdgrid, cls.stdmats)

        cls.nsnode = "test/test_inputs/nonstd.1.node"
        cls.nsele = "test/test_inputs/nonstd.1.ele"
        cls.nsmatfile = "test/test_inputs/nonstd.1.mat"
        cls.nsmats = Materials(cls.nsmatfile)
        cls.nsgrid = FEGrid(cls.nsnode, cls.nsele)
        cls.nsop = SAAF(cls.nsgrid, cls.nsmats)

        cls.smnode = "test/test_inputs/D.node"
        cls.smele = "test/test_inputs/D.ele"
        cls.smgrid = FEGrid(cls.smnode, cls.smele)
        cls.smop = SAAF(cls.smgrid, cls.mats)

        cls.symnode = "test/test_inputs/symmetric_fine.node"
        cls.symele = "test/test_inputs/symmetric_fine.ele"
        cls.symmatfile = "test/test_inputs/symmetric_fine.mat"
        cls.symmat = Materials(cls.symmatfile)
        cls.symgrid = FEGrid(cls.symnode, cls.symele)
        cls.symop = SAAF(cls.symgrid, cls.symmat)

    def symmetry_test(self):
        source = np.ones(self.fegrid.get_num_elts())
        ang_one = .5773503
        ang_two = -.5773503
        A = self.op.make_lhs(np.array([ang_one, ang_two]), 0)
        nonzero = (A!=A.transpose()).nonzero()
        ok_(np.allclose(A.A, A.transpose().A, rtol=1e-12))

    def hand_calculation_test(self):
        angles0 = np.array([.5773503, .5773503])
        A0 = self.stdop.make_lhs(angles0, 0).todense()
        hand0 = np.array([[ 0.75000007, -0.2916667,   0.,         -0.2916667],
                         [-0.2916667,   0.69245014, -0.19544165,  0.4166667 ],
                         [ 0.,         -0.19544165,  1.13490027, -0.19544165],
                         [-0.2916667,   0.4166667,  -0.19544165,  0.69245014]])
        ok_(np.allclose(A0, hand0, rtol=1e-7))
        angles1 = np.array([-.5773503, .5773503])
        A1 = self.stdop.make_lhs(angles1, 0).todense()
        hand1 = np.array([[ 0.27578343,  0.04166667,  0.        ,  0.13789172],
                          [ 0.04166667,  0.50000004,  0.04166667, -0.25000004],
                          [ 0.        ,  0.04166667,  0.27578343,  0.13789172],
                          [ 0.13789172, -0.25000004,  0.13789172,  0.88490024]])
        print(A1)
        ok_(np.allclose(A1, hand1, rtol=1e-7))
        angles2 = np.array([.5773503, -.5773503])
        A2 = self.stdop.make_lhs(angles2, 0).todense()
        hand2 = np.array([[ 0.27578343,  0.13789172,  0.        ,  0.04166667],
                          [ 0.13789172,  0.88490024,  0.13789172, -0.25000004],
                          [ 0.        ,  0.13789172,  0.27578343,  0.04166667],
                          [ 0.04166667, -0.25000004,  0.04166667,  0.50000004]])
        ok_(np.allclose(A2, hand2, rtol=1e-7))
        angles3 = np.array([-.5773503, -.5773503])
        A3 = self.stdop.make_lhs(angles3, 0).todense()
        hand3 = np.array([[ 1.13490027, -0.19544165,  0.        , -0.19544165],
                          [-0.19544165,  0.69245014, -0.2916667 ,  0.4166667 ],
                          [ 0.        , -0.2916667 ,  0.75000007, -0.2916667 ],
                          [-0.19544165,  0.4166667 , -0.2916667 ,  0.69245014]])
        ok_(np.allclose(A3, hand3, rtol=1e-7))


    # def hand_calculation_nonstd_test(self):
    #     angles = np.array([.5773503, .5773503])
    #     A = self.nsop.make_lhs(angles, 0).todense()
    #     hand = np.array([[ 0.        ,  0.        ,  0.        ,  0.        ],
    #                      [ 0.        ,  0.3849002 ,  0.1924501 ,  0.        ],
    #                      [ 0.        ,  0.1924501 ,  0.5773503 ,  0.09622505],
    #                      [ 0.        ,  0.        ,  0.09622505,  0.1924501]]) 
    #     ok_(np.allclose(A, hand, rtol=1e-7))

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

    # def balance_test(self):
    #     # Assumes one group and vacuum boundary conditions
    #     n_elts = self.fegrid.get_num_elts()
    #     n_nodes = self.fegrid.get_num_nodes()
    #     source = np.ones(n_elts)
    #     scalar_flux, ang_fluxes = self.op.solve(source, "eigenvalue", 0, "vacuum", tol=1e-3)
    #     siga = self.mats.get_siga(0, 0)
        
    #     # Calculate Total Source
    #     total_src = 0
    #     for i in range(n_elts):
    #         e = self.fegrid.element(i)
    #         total_src += self.fegrid.element_area(i)

    #     # Calculate Total Sink
    #     total_sink = 0
    #     for i in range(n_elts):
    #         e = self.fegrid.element(i)
    #         area = self.fegrid.element_area(i)
    #         vertices = e.get_vertices()
    #         phi = 0
    #         for v in vertices:
    #             phi += scalar_flux[v]
    #         phi /= 3
    #         total_sink += siga*phi*area

    #     # Calculate Total Out 
    #     # CAUTION: Only for S2
    #     ang_one = .5773503
    #     ang_two = -.5773503
    #     angles = itr.product([ang_one, ang_two], repeat=2)
    #     total_out = 0
    #     for i, ang in enumerate(angles):
    #         angle_out = 0
    #         for e in range(n_elts):
    #             # Figure out if element is on boundary
    #             vertices = self.symgrid.element(e).get_vertices()
    #             interior = -1*np.ones(3)
    #             for k, v in enumerate(vertices):
    #                 interior[k] = self.symgrid.node(v).is_interior()  
    #             if interior.sum() == 0 or interior.sum() == 1:
    #                 # Figure out what boundary we're on
    #                 # Vertex 0 & 1
    #                 if not interior[0] and not interior[1]:
    #                     normal = self.symop.assign_normal(vertices[0], vertices[1])
    #                     if type(normal) == int:
    #                         continue
    #                     if ang@normal > 0:
    #                         psi = (ang_fluxes[i, vertices[0]] + ang_fluxes[i, vertices[1]])/2
    #                         boundary_length = self.symgrid.boundary_length([vertices[0], vertices[1]], e)
    #                         angle_out += ang@normal*boundary_length*psi
    #                 # Vertex 0 & 2
    #                 if not interior[0] and not interior[2]:
    #                     normal = self.symop.assign_normal(vertices[0], vertices[2])
    #                     if type(normal) == int:
    #                         continue
    #                     if ang@normal > 0:
    #                         psi = (ang_fluxes[i, vertices[0]] + ang_fluxes[i, vertices[2]])/2
    #                         boundary_length = self.symgrid.boundary_length([vertices[0], vertices[1]], e)
    #                         angle_out += ang@normal*boundary_length*psi
    #                 # Vertex 1 & 2
    #                 if not interior[1] and not interior[2]:
    #                     normal = self.symop.assign_normal(vertices[1], vertices[2])
    #                     if type(normal) == int:
    #                         continue
    #                     if ang@normal > 0:
    #                         psi = (ang_fluxes[i, vertices[1]] + ang_fluxes[i, vertices[2]])/2
    #                         boundary_length = self.symgrid.boundary_length([vertices[0], vertices[1]], e)
    #                         angle_out += ang@normal*boundary_length*psi
    #         total_out += np.pi*angle_out
    #     eq_(total_src-total_sink, total_out)




                



                