import os
from typing import NamedTuple, Tuple

import attr
import numpy as np
import matplotlib.tri as tri

from gallo import parse

def setup_ang_quad(sn_ord):
    quad1d, solid_angle = np.polynomial.legendre.leggauss(sn_ord), 4*np.pi
    # loop over relevant polar angles
    angs = []
    weights = []
    for polar in range(int(sn_ord/2), sn_ord):
        # calculate number of points per level
        p_per_level = 4 * (sn_ord - polar)
        delta = 2.0 * np.pi / p_per_level
        # get the polar angles
        mu = quad1d[0][polar]
        # calculate point weight
        weight = quad1d[1][polar] * solid_angle / p_per_level
        # loop over azimuthal angles
        for i in range(p_per_level):
            phi = (i + 0.5) * delta
            omega = np.array([(1-mu**2.)**0.5 * np.cos(phi), (1-mu**2.)**0.5 * np.sin(phi)])
            angs.append(omega)
            weights.append(weight)
    return np.array(angs), np.array(weights)


@attr.s(slots=True, frozen=True, auto_attribs=True, repr=False)
class Element:
    el_id: int
    vertices: Tuple[int]
    mat_id: int

    def __repr__(self):
        return "{} {} {}\n".format(
        self.el_id,
        ' '.join(map(str, self.vertices)),
        self.mat_id
        )


@attr.s(slots=True, frozen=True, auto_attribs=True)
class Node:
    position: Tuple[float, float]
    id: int
    is_interior: bool


class FEGrid():
    # Discretization Orders
    num_gauss_nodes = 4
    sn_ord = 2
    angs, weights = setup_ang_quad(sn_ord)
    num_angs = len(weights)
    def __init__(self, node_file, ele_file):
        self.elts_list = parse.parse_elts(ele_file)

        self.nodes, extrema = parse.parse_nodes(node_file)
        self.xmin, self.xmax, self.ymin, self.ymax = extrema

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def num_elts(self):
        return len(self.elts_list)

    def is_corner(self, node_number):
        x, y = self.node(node_number).position
        on_xboundary = (x == self.xmax or x == self.xmin)
        on_yboundary = (y == self.ymax or y == self.ymin)
        return on_xboundary and on_yboundary

    def element(self, elt_number):
        return self.elts_list[elt_number]

    def node(self, node_or_elt_number, local_node_number=None):
        if local_node_number is None:
            return self.nodes[node_or_elt_number]
        else:
            return self.nodes[self.elts_list[node_or_elt_number].vertices[local_node_number]]

    def evaluate_basis_function(self, coefficients, point):
        # Evaluates linear basis functions of the form c1 + c2x + c3y at the point x, y
        if len(coefficients) != 3:
            raise RuntimeError(
                "Must be linear basis function with 3 coefficients")
        if len(point) != 2:
            raise RuntimeError("Must be 2D point, (x,y)")
        return coefficients[0] + coefficients[1] * point[0] + coefficients[2] * point[1]

    def gradient(self, elt_number, local_node_number):
        # WARNING: The following only works for 2D triangular elements
        element = self.elts_list[elt_number].vertices
        xbase = self.node(elt_number, local_node_number).position
        dx = np.zeros((2, 2))
        for ivert in range(2):
            other_node_number = element[(local_node_number + ivert + 1) % 3]
            dx[ivert] = self.nodes[other_node_number].position
            dx[ivert] -= xbase
        det = dx[0, 0] * dx[1, 1] - dx[1, 0] * dx[0, 1]
        retval = np.zeros(2)
        retval[0] = (-(dx[1, 1] - dx[0, 1]) / det)
        retval[1] = ((dx[1, 0] - dx[0, 0]) / det)
        return retval

    def basis(self, elt_number):
        vandermonde = np.zeros((3, 3))
        for i in range(3):
            vandermonde[i, 0] = 1
            vandermonde[i, 1] = self.node(elt_number, i).position[0]
            vandermonde[i, 2] = self.node(elt_number, i).position[1]
        coefficients = np.linalg.inv(vandermonde)
        return coefficients

    def boundary_nonzero(self, current_vert, e):
        # returns the points on the boundary where the basis function is non zero
        all_verts = np.array(self.element(e).vertices)
        vert_local_idx = np.where(all_verts == current_vert)[0][0]
        other_verts = np.delete(all_verts, vert_local_idx)

        # Get position of vertices
        posa = self.node(current_vert).position
        posb = self.node(other_verts[0]).position
        posc = self.node(other_verts[1]).position

        ab_boundary = (((posa[0] == posb[0]) and
                        (posb[0] == self.xmin or posb[0] == self.xmax))
                       or ((posa[1] == posb[1]) and
                           (posb[1] == self.ymin or posb[1] == self.ymax)))
        ac_boundary = (((posa[0] == posc[0]) and
                        (posc[0] == self.xmin or posc[0] == self.xmax))
                       or ((posa[1] == posc[1]) and
                           (posc[1] == self.ymin or posc[1] == self.ymax)))

        if ab_boundary and ac_boundary:
            return -1
        elif ab_boundary:
            verts = [current_vert, other_verts[0]]
        elif ac_boundary:
            verts = [current_vert, other_verts[1]]
        else:
            verts = [current_vert, current_vert]
        return verts

    def boundary_edges(self, boundary_vertices, e):
        # Computes the length along boundary where the basis functions are non-zero
        if boundary_vertices[0] == boundary_vertices[1]:
            verts = self.boundary_nonzero(boundary_vertices[0], e)
        else:
            verts = boundary_vertices
        points = np.zeros((2, 2))
        for i, n in enumerate(verts):
            node = self.node(n)
            points[i] = node.position
        a, b = None, None
        if points[0, 0] == points[1, 0]:
            if points[0, 1] > points[1, 1]:
                b, a = points[:, 1]
            elif points[0, 1] < points[1, 1] or points[0, 1] == points[1, 1]:
                a, b = points[:, 1]
        elif points[0, 1] == points[1, 1]:
            if points[0, 0] > points[1, 0]:
                b, a = points[:, 0]
            elif points[0, 0] < points[1, 0]:
                a, b = points[:, 0]
        if a is None and b is None:
            raise RuntimeError("Boundary Edge Error")
        else:
            return a, b

    def gauss_nodes1d(self, boundary_vertices, e):
        # Gauss nodes for 2 point quadrature on boundary
        # multiplying by the same basis function
        # Find position of boundary vertices
        a, b = self.boundary_edges(boundary_vertices, e)
        xi = 1 / np.sqrt(3)
        nodes = np.array(
            [-(b - a) / 2 * xi + (b + a) / 2, (b - a) / 2 * xi + (b + a) / 2])
        pos_n = self.node(boundary_vertices[0]).position
        pos_ns = self.node(boundary_vertices[1]).position
        gauss_nodes = np.zeros((2, 2))
        if (pos_n[0] == self.xmax and pos_ns[0] == self.xmax):
            gauss_nodes[0] = [self.xmax, nodes[0]]
            gauss_nodes[1] = [self.xmax, nodes[1]]
        elif (pos_n[0] == self.xmin and pos_ns[0] == self.xmin):
            gauss_nodes[0] = [self.xmin, nodes[0]]
            gauss_nodes[1] = [self.xmin, nodes[1]]
        elif (pos_n[1] == self.ymax and pos_ns[1] == self.ymax):
            gauss_nodes[0] = [nodes[0], self.ymax]
            gauss_nodes[1] = [nodes[1], self.ymax]
        elif (pos_n[1] == self.ymin and pos_ns[1] == self.ymin):
            gauss_nodes[0] = [nodes[0], self.ymin]
            gauss_nodes[1] = [nodes[1], self.ymin]
        return gauss_nodes


    def gauss_nodes(self, elt_number, ord=3):
        # WARNING only works for 2D triangular elements
        # http://math2.uncc.edu/~shaodeng/TEACHING/math5172/Lectures/Lect_15.PDF
        # Transform the nodes on the standard triangle to the given element
        if ord==2:
            num_gnodes = 3
            std_nodes = np.array([[0, 1/2], [1/2, 0], [1/2, 1/2]])
        if ord==3:
            num_gnodes = 4
            # Set Standard Nodes
            std_nodes = np.array([[1/3, 1/3], [1/5, 1/5], [1/5, 3/5], [3/5, 1/5]])
        elif ord==4:
            num_gnodes = 6
            std_nodes = np.array([[0.44594849091597, 0.44594849091597],
                                  [0.44594849091597, 0.10810301816807],
                                  [0.10810301816807, 0.44594849091597],
                                  [0.09157621350977, 0.09157621350977],
                                  [0.09157621350977, 0.81684757298046],
                                  [0.81684757298046, 0.09157621350977]])
        # Get nodes of element
        el_nodes = [self.node(elt_number, i) for i in [1, 2, 0]]
        pos = np.array([el_nodes[i].position for i in range(3)])
        area = self.element_area(elt_number)
        # Transformation Function
        # x(u, v) = alpha + u(beta - alpha) + v(gamma - alpha)
        trans_nodes = np.zeros((num_gnodes, 2))
        for i in range(num_gnodes):
            stdnode = std_nodes[i]
            N = np.array([[1 - stdnode[0] - stdnode[1]], [stdnode[0]], [stdnode[1]]])
            for j in range(2):
                trans_nodes[i, j] = np.sum(np.array([pos[k, j]*N[k] for k in range(3)]))
        return trans_nodes

    def element_area(self, elt_number):
        e = self.elts_list[elt_number]
        n = self.nodes[e.vertices[0]]
        xbase = n.position
        dx = np.zeros((2, 2))
        for ivert in [1, 2]:
            other_node_number = e.vertices[ivert]
            dx[ivert - 1, :] = self.nodes[other_node_number].position
            for idir in range(2):
                dx[ivert - 1, idir] -= xbase[idir]
        # WARNING: the following calculation is correct for triangles in 2D *only*.
        area = np.abs(dx[0, 0] * dx[1, 1] - dx[1, 0] * dx[0, 1]) / 2
        return area

    def gauss_quad(self, elt_number, f_values, ord=3):
        area = self.element_area(elt_number)
        if ord==2:
            integral = 1 / 3 * area * (np.sum(f_values))
        elif ord==3:
            integral = area*(-27/48*f_values[0]+25/48*np.sum(f_values[1:]))
        elif ord==4:
            integral = area*(0.22338158967801*np.sum(f_values[0:3])
                          + 0.10995174365532*np.sum(f_values[3:]))
        return integral

    def gauss_quad1d(self, f_values, boundary_vertices, e):
        # Two point Gaussian Quadrature in one dimension
        # Find length of element on boundary
        # length/2(f(-1/sqrt(3)) + f(1/sqrt(3)))
        a, b = self.boundary_edges(boundary_vertices, e)
        integral = (b - a) / 2 * (f_values[0] + f_values[1])
        return integral

    def centroid(self, elt_number):
        e = self.elts_list[elt_number]
        retval = np.zeros(2)
        for i in range(2):
            retval[i] = 0.0
        for ivert in range(3):
            n = self.nodes[e.vertices[ivert]]
            x = n.position
            for idir in range(2):
                retval[idir] += x[idir]
        for idir in range(2):
            retval[idir] /= 3
        return retval

    def assign_normal(self, nid, bid):
        pos_n = self.node(nid).position
        pos_ns = self.node(bid).position
        if (pos_n[0] == self.xmax and pos_ns[0] == self.xmax):
            normal = np.array([1, 0])
        elif (pos_n[0] == self.xmin and pos_ns[0] == self.xmin):
            normal = np.array([-1, 0])
        elif (pos_n[1] == self.ymax and pos_ns[1] == self.ymax):
            normal = np.array([0, 1])
        elif (pos_n[1] == self.ymin and pos_ns[1] == self.ymin):
            normal = np.array([0, -1])
        else:
            return -1
        return normal

    def boundary_basis_product(self, nid, bid, gauss_nodes, bn, bns, e):
        pos_n = self.node(nid).position
        pos_ns = self.node(bid).position
        if not ((pos_n[0] == self.xmax and pos_ns[0] == self.xmax)
                or (pos_n[0] == self.xmin and pos_ns[0] == self.xmin)
                or (pos_n[1] == self.ymax and pos_ns[1] == self.ymax)
                or (pos_n[1] == self.ymin and pos_ns[1] == self.ymin)):
            boundary_integral = 0
            return boundary_integral
        # Value of first basis function at boundary gauss nodes
        gn_vals = np.zeros(2)
        gn_vals[0] = self.evaluate_basis_function(bn, gauss_nodes[0])
        gn_vals[1] = self.evaluate_basis_function(bn, gauss_nodes[1])
        # Values of second basis function at boundary gauss nodes
        gns_vals = np.zeros(2)
        gns_vals[0] = self.evaluate_basis_function(bns, gauss_nodes[0])
        gns_vals[1] = self.evaluate_basis_function(bns, gauss_nodes[1])
        # Multiply basis functions together
        g_vals = gn_vals * gns_vals
        return g_vals

    def setup_triangulation(self):
        x = np.zeros(self.num_nodes)
        y = np.zeros(self.num_nodes)
        positions = (self.node(i).position for i in range(self.num_nodes))
        for i, pos in enumerate(positions):
            x[i], y[i] = pos
        # Setup triangles
        triangles = np.array([self.element(i).vertices for i in range(self.num_elts)])
        triang = tri.Triangulation(x, y, triangles=triangles)
        return triang

    def phi_at_gauss_nodes(self, triang, phi_prev, g_nodes):
        num_groups = np.shape(phi_prev)[0]
        num_nodes = np.shape(g_nodes)[0]
        phi_vals = np.zeros((num_groups, num_nodes))
        for g in range(num_groups):
            interp = tri.LinearTriInterpolator(triang, phi_prev[g])
            for i in range(num_nodes):
                phi_vals[g, i] = interp(g_nodes[i, 0], g_nodes[i, 1])
        return phi_vals
