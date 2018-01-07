from nose.tools import *
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

    def test_get_node(self):
        eq_(self.fegrid.get_node(15, 2).get_node_id(), 9, "Get Node")

    def test_get_num_elts(self):
        eq_(self.fegrid.get_num_elts(), 30)
        eq_(self.stdgrid.get_num_elts(), 2)

    def test_get_num_nodes(self):
        eq_(self.fegrid.get_num_nodes(), 19)
        eq_(self.stdgrid.get_num_nodes(), 4)

    def test_num_interior(self):
        eq_(self.fegrid.get_num_interior_nodes(), 13, "interior nodes")
        eq_(self.stdgrid.get_num_interior_nodes(), 0)

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

    def test_quad(self):
        fvals = np.array([1, 1, 1])
        fvals2 = np.array([2, 2, 2])
        eq_(self.stdgrid.gauss_quad(0, fvals), .5, "quadrature")
        eq_(self.stdgrid.gauss_quad(1, fvals), .5)
        eq_(self.stdgrid.gauss_quad(0, fvals2), 1)

    def test_centroid(self):
        eq_(self.stdgrid.centroid(0)[0], 1/3)
        eq_(self.stdgrid.centroid(0)[1], 1/3)


