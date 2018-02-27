from nose.tools import *
from numpy.testing import *
import numpy as np
import sys
sys.path.append('../src')

from fe import *

class TestFe:
    @classmethod
    def setup_class(cls):
        cls.nodefile = "test/test_inputs/test.node"
        cls.elefile = "test/test_inputs/test.ele"
        cls.fegrid = FEGrid(cls.nodefile, cls.elefile)

        cls.stdnode = "test/test_inputs/std.1.node"
        cls.stdele = "test/test_inputs/std.1.ele"
        cls.stdgrid = FEGrid(cls.stdnode, cls.stdele)

    def test_get_boundary(self):
        eq_(self.fegrid.get_boundary("xmin"), 0)
        eq_(self.fegrid.get_boundary("xmax"), 1)
        eq_(self.fegrid.get_boundary("ymin"), 0)
        eq_(self.fegrid.get_boundary("ymax"), 1)

    def test_is_corner(self):
        eq_(self.fegrid.is_corner(0), True)
        eq_(self.fegrid.is_corner(1), True)
        eq_(self.fegrid.is_corner(2), True)
        eq_(self.fegrid.is_corner(3), True)
        eq_(self.fegrid.is_corner(4), False)
        eq_(self.fegrid.is_corner(5), False)
        eq_(self.fegrid.is_corner(6), False)
        eq_(self.fegrid.is_corner(7), False)
        eq_(self.fegrid.is_corner(8), False)
        eq_(self.fegrid.is_corner(9), False)
        eq_(self.fegrid.is_corner(10), False)
        eq_(self.fegrid.is_corner(11), False)
        eq_(self.fegrid.is_corner(12), False)
        eq_(self.fegrid.is_corner(13), False)
        eq_(self.fegrid.is_corner(14), False)
        eq_(self.fegrid.is_corner(15), False)
        eq_(self.fegrid.is_corner(16), False)
        eq_(self.fegrid.is_corner(17), False)
        eq_(self.fegrid.is_corner(18), False)

    def test_get_node(self):
        eq_(self.fegrid.get_node(15, 2).get_node_id(), 9, "Get Node")

    def test_get_num_elts(self):
        eq_(self.fegrid.get_num_elts(), 30)
        eq_(self.stdgrid.get_num_elts(), 2)

    def test_get_num_nodes(self):
        eq_(self.fegrid.get_num_nodes(), 19)
        eq_(self.stdgrid.get_num_nodes(), 4)

    def test_num_interior_nodes(self):
        eq_(self.fegrid.get_num_interior_nodes(), 13, "interior nodes")
        eq_(self.stdgrid.get_num_interior_nodes(), 0)

    def test_get_mat_id(self):
        eq_(self.fegrid.get_mat_id(0), 0)
        eq_(self.fegrid.get_mat_id(21), 1)
        eq_(self.fegrid.get_mat_id(17), 1)
        eq_(self.fegrid.get_mat_id(13), 0)

    def test_evaluate_basis_function(self):
        eq_(self.fegrid.evaluate_basis_function([1, 0, 0], [3, 2]), 1)
        eq_(self.fegrid.evaluate_basis_function([1, 1, 1], [4, 5]), 10)
        eq_(self.fegrid.evaluate_basis_function([2, 0, 2], [6, 1]), 4)
        eq_(self.fegrid.evaluate_basis_function([0, 1, 1], [0, 12]), 12)

    def test_gradient(self):
        num_ele = self.fegrid.get_num_elts()
        for e in range(num_ele):
            coef = self.fegrid.basis(e)
            for n in range(3):
                bn = coef[:, n]
                grad = [bn[1], bn[2]]
                dx, dy = self.fegrid.gradient(e, n)
                assert_allclose(bn[1], dx)
                assert_allclose(bn[2], dy)

    def test_basis(self):
        V = np.array([[1, 0, 1],
                      [1, 0, 0],
                      [1, 1, 0]])
        C = np.linalg.inv(V)
        assert_array_equal(C, self.stdgrid.basis(0))

    def test_boundary_length(self):
        eq_(self.fegrid.boundary_length([0, 5]), .5)
        eq_(self.fegrid.boundary_length([1, 5]), .5)
        eq_(self.fegrid.boundary_length([12, 2]), .5)

    def test_gauss_nodes1d(self):
        nodes_half = np.array([(-.25*1/np.sqrt(3) + .25),(.25*1/np.sqrt(3) + .25)])
        assert_array_equal(self.fegrid.gauss_nodes1d([0, 5]), nodes_half)
        assert_array_equal(self.fegrid.gauss_nodes1d([1, 5]), nodes_half)
        
    def test_gauss_nodes(self):
        C = self.stdgrid.gauss_nodes(0)
        eq_(C[0, 0], 0, "node x1")
        eq_(C[0, 1], .5, "node y1")
        eq_(C[1, 0], .5, "node x2")
        eq_(C[1, 1], 0, "node y2")
        eq_(C[2, 0], .5, "node x3")
        eq_(C[2, 1], .5, "node y3")

    def test_area(self):
        eq_(self.stdgrid.element_area(0), .5, "element_area")
        eq_(self.stdgrid.element_area(1), .5, "element_area")

    def test_quad1d(self):
        fvals = [1, 1, 1]
        eq_(self.fegrid.gauss_quad1d(fvals, [1, 5]), .5)
        eq_(self.fegrid.gauss_quad1d(fvals, [0, 5]), .5)
        
    def test_quad(self):
        fvals = np.array([1, 1, 1])
        fvals2 = np.array([2, 2, 2])
        eq_(self.stdgrid.gauss_quad(0, fvals), .5, "quadrature")
        eq_(self.stdgrid.gauss_quad(1, fvals), .5)
        eq_(self.stdgrid.gauss_quad(0, fvals2), 1)

    def test_centroid(self):
        eq_(self.stdgrid.centroid(0)[0], 1/3)
        eq_(self.stdgrid.centroid(0)[1], 1/3)


