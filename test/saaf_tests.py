from nose.tools import *
from numpy.testing import *
from nose.plugins.attrib import attr
import numpy as np
import itertools as itr

from gallo.formulations.saaf import SAAF
from gallo.fe import FEGrid
from gallo.materials import Materials
from gallo.solvers import Solver

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
        cls.solv2 = Solver(cls.op2)

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
        cls.symmatfile = "test/test_inputs/noscatter.mat"
        cls.symmat = Materials(cls.symmatfile)
        cls.symgrid = FEGrid(cls.symnode, cls.symele, sn_ord=2)
        cls.symop = SAAF(cls.symgrid, cls.symmat)
        cls.symsolv = Solver(cls.symop)

        cls.scatmatfile = "test/test_inputs/scattering1g.mat"
        cls.scatmat = Materials(cls.scatmatfile)
        cls.scatop = SAAF(cls.stdgrid, cls.scatmat)
        cls.scat3op = SAAF(cls.std3grid, cls.scatmat)

        cls.oneop = SAAF(cls.symgrid, cls.scatmat)
        cls.onesolv = Solver(cls.oneop)

        cls.orignode = "test/test_inputs/origin_centered10_fine.node"
        cls.origele = "test/test_inputs/origin_centered10_fine.ele"
        cls.origrid = FEGrid(cls.orignode, cls.origele)
        cls.twomatfile = "test/test_inputs/scattering2g.mat"
        cls.twoscatmat = Materials(cls.twomatfile)
        cls.twop = SAAF(cls.origrid, cls.twoscatmat)
        cls.twosolv = Solver(cls.twop)

        cls.fissionmatfile = "test/test_inputs/fissiontest.mat"
        cls.fissionmat = Materials(cls.fissionmatfile)
        cls.fissionop = SAAF(cls.symgrid, cls.fissionmat)
        cls.fissolv = Solver(cls.fissionop)

    def symmetry_test(self):
        source = np.ones(self.fegrid.num_elts)
        ang_one = .5773503
        ang_two = -.5773503
        A = self.op.make_lhs(np.array([ang_one, ang_two]), 0)
        ok_(np.allclose(A.A, A.transpose().A, rtol=1e-12))

    def test_eigenvalue(self):
        source = np.zeros((self.fissionop.num_groups, self.fissionop.num_elts))
        fluxes = self.fissolv.solve(source, eigenvalue=True)
        k = fluxes['k']
        assert_allclose(k, 0.234582, rtol=0.5)

    # def hand_calculation_lhs_test(self):
    #     angles0 = np.array([.5773503, .5773503])
    #     A0 = self.stdop.make_lhs(angles0, 0).todense()
    #     hand0 = np.array([[ 0.75000007, -0.2916667,   0.,         -0.2916667],
    #                      [-0.2916667,   0.69245014, -0.19544165,  0.4166667 ],
    #                      [ 0.,         -0.19544165,  1.13490027, -0.19544165],
    #                      [-0.2916667,   0.4166667,  -0.19544165,  0.69245014]])
    #     ok_(np.allclose(A0, hand0, rtol=1e-7))
    #     angles1 = np.array([-.5773503, .5773503])
    #     A1 = self.stdop.make_lhs(angles1, 0).todense()
    #     hand1 = np.array([[ 0.27578343,  0.04166667,  0.        ,  0.13789172],
    #                       [ 0.04166667,  0.50000004,  0.04166667, -0.25000004],
    #                       [ 0.        ,  0.04166667,  0.27578343,  0.13789172],
    #                       [ 0.13789172, -0.25000004,  0.13789172,  0.88490024]])
    #     ok_(np.allclose(A1, hand1, rtol=1e-7))
    #     angles2 = np.array([.5773503, -.5773503])
    #     A2 = self.stdop.make_lhs(angles2, 0).todense()
    #     hand2 = np.array([[ 0.27578343,  0.13789172,  0.        ,  0.04166667],
    #                       [ 0.13789172,  0.88490024,  0.13789172, -0.25000004],
    #                       [ 0.        ,  0.13789172,  0.27578343,  0.04166667],
    #                       [ 0.04166667, -0.25000004,  0.04166667,  0.50000004]])
    #     ok_(np.allclose(A2, hand2, rtol=1e-7))
    #     angles3 = np.array([-.5773503, -.5773503])
    #     A3 = self.stdop.make_lhs(angles3, 0).todense()
    #     hand3 = np.array([[ 1.13490027, -0.19544165,  0.        , -0.19544165],
    #                       [-0.19544165,  0.69245014, -0.2916667 ,  0.4166667 ],
    #                       [ 0.        , -0.2916667 ,  0.75000007, -0.2916667 ],
    #                       [-0.19544165,  0.4166667 , -0.2916667 ,  0.69245014]])
    #     ok_(np.allclose(A3, hand3, rtol=1e-7))
    #
    # def hand_calculation_rhs_test(self):
    #     q = np.ones((1, 4))
    #     phi_prev = np.zeros((1, 4))
    #     angles0 = np.array([.5773503, .5773503])
    #     b0 = self.stdop.make_rhs(0, q, angles0, 0, phi_prev=phi_prev)
    #     hand0 = np.array([-0.03268117,  0.02652582,  0.05920699,  0.02652582])
    #     ok_(np.allclose(b0, hand0, rtol=1e-7))
    #     angles1 = np.array([-.5773503, .5773503])
    #     b1 = self.stdop.make_rhs(0, q, angles1, 1, phi_prev=phi_prev)
    #     hand1 = np.array([ 0.01326291, -0.01941825,  0.01326291,  0.0724699])
    #     ok_(np.allclose(b1, hand1, rtol=1e-7))
    #     angles2 = np.array([.5773503, -.5773503])
    #     b2 = self.stdop.make_rhs(0, q, angles2, 2, phi_prev=phi_prev)
    #     hand2 = np.array([ 0.01326291,  0.0724699 ,  0.01326291, -0.01941825])
    #     ok_(np.allclose(b2, hand2, rtol=1e-7))
    #     angles3 = np.array([-.5773503, -.5773503])
    #     b3 = self.stdop.make_rhs(0, q, angles3, 3, phi_prev=phi_prev)
    #     hand3 = np.array([ 0.05920699,  0.02652582, -0.03268117,  0.02652582])
    #     ok_(np.allclose(b3, hand3, rtol=1e-7))

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

    @attr('slow')
    def test_no_scat(self):
        source = 10*np.ones((self.symop.num_groups, self.symop.num_elts))
        fluxes = self.symsolv.solve(source)
        phis = fluxes['Phi']
        gold_phis = np.array([np.loadtxt("test/test_outputs/saaf_no_scat.out")])
        assert_array_almost_equal(phis, gold_phis, decimal=4)

    @attr('slow')
    def test_1g(self):
        source = 10*np.ones((self.oneop.num_groups, self.oneop.num_elts))
        fluxes = self.onesolv.solve(source)
        phis = fluxes['Phi']
        gold_phis = np.array([np.loadtxt("test/test_outputs/saaf1g.out")])
        assert_array_almost_equal(phis, gold_phis, decimal=4)

    @attr('slow')
    def test_2g(self):
        source = 10*np.ones((self.twop.num_groups, self.twop.num_elts))
        fluxes = self.twosolv.solve(source)
        phis = fluxes['Phi']
        gold_phis = np.loadtxt("test/test_outputs/saaf2g.out")
        assert_array_almost_equal(phis, gold_phis, decimal=4)

    # @attr('slow')
    # def balance_test(self):
        # # Assumes one group and vacuum boundary conditions
        # n_elts = self.symgrid.num_elts
        # n_nodes = self.symgrid.num_nodes
        # source = 10*np.ones((self.symmat.num_groups, n_elts))
        # scalar_flux, ang_fluxes = self.symsolv.solve(source)
        # siga = self.symmat.get_siga(0, 0)
        #
        # # Calculate Total Source
        # total_src = 0
        # for i in range(n_elts):
        #     e = self.symgrid.element(i)
        #     total_src += 10*self.symgrid.element_area(i)
        #
        # # Calculate Total Sink
        # total_sink = 0
        # for i in range(n_elts):
        #     e = self.symgrid.element(i)
        #     area = self.symgrid.element_area(i)
        #     vertices = e.vertices
        #     phi = 0
        #     for v in vertices:
        #         phi += scalar_flux[0, v]
        #     phi /= 3
        #     total_sink += siga*phi*area
        #
        # # Calculate Total Out
        # # CAUTION: Only for S2
        # ang_one = .5773503
        # ang_two = -.5773503
        # angles = itr.product([ang_one, ang_two], repeat=2)
        # total_out = 0
        # for i, ang in enumerate(angles):
        #     angle_out = 0
        #     for e in range(n_elts):
        #         # Figure out if element is on boundary
        #         vertices = self.symgrid.element(e).vertices
        #         interior = np.array([self.symgrid.node(v).is_interior for v in vertices])
        #         print(interior)
        #         if interior.sum() <= 1:
        #             # Figure out what boundary we're on
        #             for idx in [[0, 1], [0, 2], [1, 2]]:
        #                 m, n = idx
        #                 if interior[m] + interior[n] == 0:
        #                     normal = self.symgrid.assign_normal(vertices[m], vertices[n])
        #                     if type(normal) == int:
        #                         continue
        #                     if ang@normal > 0:
        #                         psi = (ang_fluxes[0, i, vertices[m]] + ang_fluxes[0, i, vertices[n]])/2
        #                         a, b = self.symgrid.boundary_edges([vertices[m], vertices[n]], e)
        #                         boundary_length = np.abs(b - a)
        #                         #angle_out += ang@normal*boundary_length*psi
        #                         angle_out += boundary_length*psi
        #     total_out += self.symgrid.weights[0]*angle_out
        #
        # assert_almost_equals((total_src-(total_sink + total_out))/total_src, 0, places=6)
        # Assumes one group and vacuum boundary conditions
        # n_elts = self.fe2grid.num_elts
        # n_nodes = self.fe2grid.num_nodes
        # source = np.ones((self.mats2.num_groups, n_elts))
        # scalar_flux, ang_fluxes = self.solv2.solve(source)
        # siga = self.mats2.get_siga(0, 0)
        #
        # # Calculate Total Source
        # total_src = 0
        # for i in range(n_elts):
        #     e = self.fe2grid.element(i)
        #     total_src += self.fe2grid.element_area(i)
        #
        # # Calculate Total Sink
        # total_sink = 0
        # for i in range(n_elts):
        #     e = self.fe2grid.element(i)
        #     area = self.fe2grid.element_area(i)
        #     vertices = e.vertices
        #     phi = 0
        #     for v in vertices:
        #         phi += scalar_flux[0, v]
        #     phi /= 3
        #     total_sink += siga*phi*area
        #
        # # Calculate Total Out
        # # CAUTION: Only for S2
        # ang_one = .5773503
        # ang_two = -.5773503
        # angles = itr.product([ang_one, ang_two], repeat=2)
        # total_out = 0
        # for i, ang in enumerate(angles):
        #     angle_out = 0
        #     for e in range(n_elts):
        #         # Figure out if element is on boundary
        #         vertices = self.fe2grid.element(e).vertices
        #         interior = -1*np.ones(3)
        #         for k, v in enumerate(vertices):
        #             interior[k] = self.fe2grid.node(v).is_interior
        #         if interior.sum() == 0 or interior.sum() == 1:
        #             # Figure out what boundary we're on
        #             for idx in [[0, 1], [0, 2], [1, 2]]:
        #                 m = idx[0]
        #                 n = idx[1]
        #                 if not interior[m] and not interior[n]:
        #                     normal = self.fe2grid.assign_normal(vertices[m], vertices[n])
        #                     if type(normal) == int:
        #                         continue
        #                     if ang@normal > 0:
        #                         psi = (ang_fluxes[0, i, vertices[m]] + ang_fluxes[0, i, vertices[n]])/2
        #                         a, b = self.fe2grid.boundary_edges([vertices[m], vertices[n]], e)
        #                         boundary_length = b - a
        #                         angle_out += ang@normal*boundary_length*psi
        #     total_out += np.pi*angle_out
        # assert_almost_equals((np.abs(total_src-total_sink) - total_out)/total_src, 0, places=6)
