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
        cls.matfile = "test/test_inputs/mesh0_test.mat"
        cls.fegrid = FEGrid(cls.nodefile, cls.elefile)
        cls.mats = Materials(cls.matfile)
        cls.op = SAAF(cls.fegrid, cls.mats)

        cls.node2file = "test/test_inputs/mesh2.node"
        cls.ele2file = "test/test_inputs/mesh2.ele"
        cls.mat2file = "test/test_inputs/mesh2_test.mat"
        cls.fe2grid = FEGrid(cls.node2file, cls.ele2file)
        cls.mats2 = Materials(cls.mat2file)
        cls.op2 = SAAF(cls.fe2grid, cls.mats2)

        cls.stdnode = "test/test_inputs/std.node"
        cls.stdele = "test/test_inputs/std.ele"
        cls.stdmatfile = "test/test_inputs/std_test.mat"
        cls.stdmats = Materials(cls.stdmatfile)
        cls.stdgrid = FEGrid(cls.stdnode, cls.stdele)
        cls.stdop = SAAF(cls.stdgrid, cls.stdmats)

        cls.std3node = "test/test_inputs/std3.node"
        cls.std3ele = "test/test_inputs/std3.ele"
        cls.std3matfile = "test/test_inputs/std3_test.mat"
        cls.std3mats = Materials(cls.std3matfile)
        cls.std3grid = FEGrid(cls.std3node, cls.std3ele)
        cls.std3op = SAAF(cls.std3grid, cls.std3mats)

        cls.nsnode = "test/test_inputs/nonstd.1.node"
        cls.nsele = "test/test_inputs/nonstd.1.ele"
        cls.nsmatfile = "test/test_inputs/nonstd_test.mat"
        cls.nsmats = Materials(cls.nsmatfile)
        cls.nsgrid = FEGrid(cls.nsnode, cls.nsele)
        cls.nsop = SAAF(cls.nsgrid, cls.nsmats)

        cls.smnode = "test/test_inputs/D.node"
        cls.smele = "test/test_inputs/D.ele"
        cls.smgrid = FEGrid(cls.smnode, cls.smele)
        cls.smop = SAAF(cls.smgrid, cls.mats)

        cls.symnode = "test/test_inputs/symmetric_fine.node"
        cls.symele = "test/test_inputs/symmetric_fine.ele"
        cls.symmatfile = "test/test_inputs/symmetricfine_test.mat"
        cls.symmat = Materials(cls.symmatfile)
        cls.symgrid = FEGrid(cls.symnode, cls.symele)
        cls.symop = SAAF(cls.symgrid, cls.symmat)

        cls.scatmatfile = "test/test_inputs/scattering1g.mat"
        cls.scatmat = Materials(cls.scatmatfile)
        cls.scatop = SAAF(cls.stdgrid, cls.scatmat)
        cls.scat3op = SAAF(cls.std3grid, cls.scatmat)

        cls.fissionmatfile = "test/test_inputs/fissiontest.mat"
        cls.fissionmat = Materials(cls.fissionmatfile)
        cls.fissionop = SAAF(cls.stdgrid, cls.fissionmat)

    def symmetry_test(self):
        source = np.ones(self.fegrid.get_num_elts())
        ang_one = .5773503
        ang_two = -.5773503
        A = self.op.make_lhs(np.array([ang_one, ang_two]), 0)
        ok_(np.allclose(A.A, A.transpose().A, rtol=1e-12))

    def hand_calculation_lhs_test(self):
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

    def hand_calculation_rhs_test(self):
        q = np.ones((1, 4))
        phi_prev = np.zeros((1, 4))
        angles0 = np.array([.5773503, .5773503])
        b0 = self.stdop.make_rhs(0, q, angles0, 0, phi_prev=phi_prev)
        hand0 = np.array([-0.03268117,  0.02652582,  0.05920699,  0.02652582])
        ok_(np.allclose(b0, hand0, rtol=1e-7))
        angles1 = np.array([-.5773503, .5773503])
        b1 = self.stdop.make_rhs(0, q, angles1, 1, phi_prev=phi_prev)
        hand1 = np.array([ 0.01326291, -0.01941825,  0.01326291,  0.0724699])
        ok_(np.allclose(b1, hand1, rtol=1e-7))
        angles2 = np.array([.5773503, -.5773503])
        b2 = self.stdop.make_rhs(0, q, angles2, 2, phi_prev=phi_prev)
        hand2 = np.array([ 0.01326291,  0.0724699 ,  0.01326291, -0.01941825])
        ok_(np.allclose(b2, hand2, rtol=1e-7))
        angles3 = np.array([-.5773503, -.5773503])
        b3 = self.stdop.make_rhs(0, q, angles3, 3, phi_prev=phi_prev)
        hand3 = np.array([ 0.05920699,  0.02652582, -0.03268117,  0.02652582])
        ok_(np.allclose(b3, hand3, rtol=1e-7))

    def hand_calculation_8cell_test(self):
        angles0 = np.array([.5773503, .5773503])
        A0 = self.std3op.make_lhs(angles0, 0).todense()
        hand0 = np.array([[ 0.37500004,  0.        ,  0.        ,  0.        , -0.31250004,
                             0.01041667,  0.        ,  0.        ,  0.01041667],
                           [ 0.        ,  0.47122509,  0.        ,  0.        ,  0.3541667 ,
                             0.        ,  0.        , -0.27480418, -0.3229167 ],
                           [ 0.        ,  0.        ,  0.56745014,  0.        , -0.31250004,
                             0.        ,  0.05852919,  0.05852919,  0.        ],
                           [ 0.        ,  0.        ,  0.        ,  0.47122509,  0.3541667 ,
                            -0.3229167 , -0.27480418,  0.        ,  0.        ],
                           [-0.31250004,  0.3541667 , -0.31250004,  0.3541667 ,  1.50000014,
                            -0.31250004, -0.31250004, -0.31250004, -0.31250004],
                           [ 0.01041667,  0.        ,  0.        , -0.3229167 , -0.31250004,
                             0.7083334 ,  0.        ,  0.        ,  0.        ],
                           [ 0.        ,  0.        ,  0.05852919, -0.27480418, -0.31250004,
                             0.        ,  0.9007835 ,  0.        ,  0.        ],
                           [ 0.        , -0.27480418,  0.05852919,  0.        , -0.31250004,
                             0.        ,  0.        ,  0.9007835 ,  0.        ],
                           [ 0.01041667, -0.3229167 ,  0.        ,  0.        , -0.31250004,
                             0.        ,  0.        ,  0.        ,  0.7083334 ]])
        ok_(np.allclose(A0, hand0, rtol=1e-7))

    @nottest
    # Slow test, only enable when necessary
    def balance_test(self):
        # Assumes one group and vacuum boundary conditions
        n_elts = self.symgrid.get_num_elts()
        n_nodes = self.symgrid.get_num_nodes()
        source = np.ones(n_elts)
        scalar_flux, ang_fluxes = self.symop.solve_outer(source, tol=1e-3)
        siga = self.symmat.get_siga(0, 0)

        # Calculate Total Source
        total_src = 0
        for i in range(n_elts):
            e = self.symgrid.element(i)
            total_src += self.symgrid.element_area(i)

        # Calculate Total Sink
        total_sink = 0
        for i in range(n_elts):
            e = self.symgrid.element(i)
            area = self.symgrid.element_area(i)
            vertices = e.get_vertices()
            phi = 0
            for v in vertices:
                phi += scalar_flux[v]
            phi /= 3
            total_sink += siga*phi*area

        # Calculate Total Out
        # CAUTION: Only for S2
        ang_one = .5773503
        ang_two = -.5773503
        angles = itr.product([ang_one, ang_two], repeat=2)
        total_out = 0
        for i, ang in enumerate(angles):
            angle_out = 0
            for e in range(n_elts):
                # Figure out if element is on boundary
                vertices = self.symgrid.element(e).get_vertices()
                interior = -1*np.ones(3)
                for k, v in enumerate(vertices):
                    interior[k] = self.symgrid.node(v).is_interior()
                if interior.sum() == 0 or interior.sum() == 1:
                    # Figure out what boundary we're on
                    for idx in [[0, 1], [0, 2], [1, 2]]:
                        m = idx[0]
                        n = idx[1]
                        if not interior[m] and not interior[n]:
                            normal = self.symop.assign_normal(vertices[m], vertices[n])
                            if type(normal) == int:
                                continue
                            if ang@normal > 0:
                                psi = (ang_fluxes[i, vertices[m]] + ang_fluxes[i, vertices[n]])/2
                                a, b = self.symgrid.boundary_edges([vertices[m], vertices[n]], e)
                                boundary_length = b - a
                                angle_out += ang@normal*boundary_length*psi
            total_out += np.pi*angle_out
        assert_almost_equals((np.abs(total_src-total_sink) - total_out)/total_src, 0, places=6)
        # Assumes one group and vacuum boundary conditions
        n_elts = self.fe2grid.get_num_elts()
        n_nodes = self.fe2grid.get_num_nodes()
        source = np.ones(n_elts)
        scalar_flux, ang_fluxes = self.op2.solve(source, "eigenvalue", 0, tol=1e-5)
        siga = self.mats2.get_siga(0, 0)

        # Calculate Total Source
        total_src = 0
        for i in range(n_elts):
            e = self.fe2grid.element(i)
            total_src += self.fe2grid.element_area(i)

        # Calculate Total Sink
        total_sink = 0
        for i in range(n_elts):
            e = self.fe2grid.element(i)
            area = self.fe2grid.element_area(i)
            vertices = e.get_vertices()
            phi = 0
            for v in vertices:
                phi += scalar_flux[v]
            phi /= 3
            total_sink += siga*phi*area

        # Calculate Total Out
        # CAUTION: Only for S2
        ang_one = .5773503
        ang_two = -.5773503
        angles = itr.product([ang_one, ang_two], repeat=2)
        total_out = 0
        for i, ang in enumerate(angles):
            angle_out = 0
            for e in range(n_elts):
                # Figure out if element is on boundary
                vertices = self.fe2grid.element(e).get_vertices()
                interior = -1*np.ones(3)
                for k, v in enumerate(vertices):
                    interior[k] = self.fe2grid.node(v).is_interior()
                if interior.sum() == 0 or interior.sum() == 1:
                    # Figure out what boundary we're on
                    for idx in [[0, 1], [0, 2], [1, 2]]:
                        m = idx[0]
                        n = idx[1]
                        if not interior[m] and not interior[n]:
                            normal = self.op2.assign_normal(vertices[m], vertices[n])
                            if type(normal) == int:
                                continue
                            if ang@normal > 0:
                                psi = (ang_fluxes[i, vertices[m]] + ang_fluxes[i, vertices[n]])/2
                                a, b = self.fe2grid.boundary_edges([vertices[m], vertices[n]], e)
                                boundary_length = b - a
                                angle_out += ang@normal*boundary_length*psi
            total_out += np.pi*angle_out
        assert_almost_equals((np.abs(total_src-total_sink) - total_out)/total_src, 0, places=6)
