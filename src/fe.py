import os
import numpy as np
import sys

class Element():
    def __init__(self, el_id, vertices, mat_id):
        """ Constructor for a triangular element in 2D """
        self.el_id = el_id
        self.vertices = vertices
        self.mat_id = mat_id

    def __getitem__(self, index):
        return self.vertices[index]

    def get_el_id(self):
        return self.el_id

    def get_vertices(self):
        return self.vertices

    def get_mat_id(self):
        return self.mat_id

class Node():
    def __init__(self, position, node_id, interior_node_id, isinterior):
        self.position = position
        self.id = node_id
        self.isinterior = isinterior
        self.interior_node_id = interior_node_id

    def get_position(self):
        return self.position

    def get_node_id(self):
        return self.id

    def get_interior_node_id(self):
        return self.interior_node_id

    def is_interior(self):
        return self.isinterior

class FEGrid():
    def __init__(self, node_file, ele_file):
        # verify files exists
        assert os.path.exists(node_file), "Node file: " + node_file\
            + " does not exist"
        assert os.path.exists(ele_file), "Ele file: " + ele_file\
            + " does not exist"
        
        with open(node_file) as nf:
            line = nf.readline()
            data = line.split(" ")
            self.num_nodes = int(data[0])
            self.nodes = []
            self.interior_nodes = []
            self.num_interior_nodes = 0
            for i in range(self.num_nodes):
                line = nf.readline()
                #remove extra whitespace
                line = ' '.join(line.split())
                data = line.split(" ")
                node_id, x, y, boundary = data
                x = float(x)
                y = float(y)
                # Set boundary data 
                if i==0:
                    self.xmin = x
                    self.ymin = y
                elif i==2:
                    self.xmax = x
                    self.ymax = y
                is_interior = not int(boundary)
                interior_node_id = -1
                if is_interior: 
                    interior_node_id = self.num_interior_nodes
                    self.num_interior_nodes += 1
                    self.interior_nodes.append(Node([x, y], int(node_id), interior_node_id, is_interior))
                self.nodes.append(Node([x, y], int(node_id), interior_node_id, is_interior))

        with open(ele_file) as ef:
            line = ef.readline()
            data = line.split(" ")
            self.num_ele = int(data[0])
            self.elts = []
            for i in range(self.num_ele):
                line = ef.readline()
                #remove extra whitespace
                line = ' '.join(line.split())
                data = line.split(" ")
                el_id, *vertices, mat_id = data
                vertices = [int(vert) for vert in vertices]
                self.elts.append(Element(int(el_id), vertices, int(mat_id)))

    def get_boundary(self, which_boundary):
        if which_boundary=="xmin":
            return self.xmin
        elif which_boundary=="ymin":
            return self.ymin
        elif which_boundary=="xmax":
            return self.xmax
        elif which_boundary=="ymax":
            return self.ymax
        else:
            raise RuntimeError("Boundary must be xmin, ymin, xmax, or ymax")
    
    def is_corner(self, node_number):
        x, y = self.node(node_number).get_position()
        on_xboundary = (x == self.xmax or x == self.xmin)
        on_yboundary = (y == self.ymax or y == self.ymin)
        if on_xboundary and on_yboundary:
            return True
        else:
            return False

    def get_node(self, elt_number, local_node_number):
        return self.nodes[self.elts[elt_number][local_node_number]]

    def get_num_elts(self):
        return np.size(self.elts)

    def get_num_nodes(self):
        return np.size(self.nodes)

    def get_num_interior_nodes(self):
        return self.num_interior_nodes

    def element(self, elt_number):
        return self.elts[elt_number]

    def node(self, node_number):
        return self.nodes[node_number]

    # def interior_node(self, interior_node_number):
    #     return self.interior_nodes[interior_node_number]

    def get_mat_id(self, elt_number):
        return self.element(elt_number).get_mat_id()

    def evaluate_basis_function(self, coefficients, point):
        # Evaluates linear basis functions of the form c1 + c2x + c3y at the point x, y
        if len(coefficients) != 3:
            raise RuntimeError("Must be linear basis function with 3 coefficients")
        if len(point) != 2:
            raise RuntimeError("Must be 2D point, (x,y)")
        return coefficients[0] + coefficients[1]*point[0] + coefficients[2]*point[1]

    def gradient(self, elt_number, local_node_number):
        # WARNING: The following only works for 2D triangular elements
        e = self.elts[elt_number].get_vertices()
        n = self.get_node(elt_number, local_node_number)
        xbase = n.get_position()
        dx = np.zeros((2, 2))
        for ivert in range(2):
            other_node_number = e[(local_node_number + ivert + 1)%3]
            dx[ivert] = self.nodes[other_node_number].get_position()
            dx[ivert] -= xbase
        det = dx[0, 0]*dx[1, 1] - dx[1, 0]*dx[0, 1]
        retval = np.zeros(2)
        retval[0] = (-(dx[1, 1] - dx[0, 1])/det)
        retval[1] = ((dx[1, 0] - dx[0, 0])/det)
        return retval

    def basis(self, elt_number):
        V = np.zeros((3, 3))
        for i in range(3):
            V[i, 0] = 1
            V[i, 1] = self.get_node(elt_number, i).get_position()[0]
            V[i, 2] = self.get_node(elt_number, i).get_position()[1]
        C = np.linalg.inv(V)
        return C

    def boundary_nonzero(self, current_vert, e):
        # returns the points on the boundary where the basis function is non zero
        all_verts = np.array(self.element(e).get_vertices())
        vert_local_idx = np.where(all_verts == current_vert)[0][0]
        other_verts = np.delete(all_verts, vert_local_idx)

        # Get position of vertices
        posa = self.node(current_vert).get_position()
        posb = self.node(other_verts[0]).get_position()
        posc = self.node(other_verts[1]).get_position()
        
        
        ab_boundary = (((posa[0] == posb[0]) and (posb[0] == self.xmin or posb[0] == self.xmax)) 
                    or ((posa[1] == posb[1]) and (posb[1] == self.ymin or posb[1] == self.ymax)))
        ac_boundary = (((posa[0] == posc[0]) and (posc[0] == self.xmin or posc[0] == self.xmax)) 
                    or ((posa[1] == posc[1]) and (posc[1] == self.ymin or posc[1] == self.ymax)))

        if ab_boundary and ac_boundary:
            return -1
        elif ab_boundary:
            verts = [current_vert, other_verts[0]]
        elif ac_boundary:
            verts = [current_vert, other_verts[1]]
        else:
            verts = [current_vert, current_vert]
        return verts

    def boundary_length(self, boundary_vertices, e):
        # Computes the length along boundary where the basis functions are non-zero
        if boundary_vertices[0] == boundary_vertices[1]:
            verts = self.boundary_nonzero(boundary_vertices[0], e)
        else:
            verts = boundary_vertices
        points = np.zeros((2, 2))
        for i, n in enumerate(verts):
            node = self.node(n)
            points[i] = node.get_position()
        length = np.max(np.abs(points[0] - points[1]))
        return length 

    def gauss_nodes1d(self, boundary_vertices, e):
        # Gauss nodes for 2 point quadrature on boundary
        #multiplying by the same basis function
        length = self.boundary_length(boundary_vertices, e)
        xi = 1/np.sqrt(3)
        half_length = length/2
        nodes = np.array([-half_length*xi + half_length, half_length*xi + half_length])
        return nodes

    def gauss_nodes(self, elt_number):
        # WARNING only works for 2D triangular elements
        # Using second order Gaussian Quadrature nodes (0, .5), (.5, 0), (.5, .5)
        # Transform the nodes on the standard triangle to the given element

        # Get nodes of element
        alpha = self.get_node(elt_number, 1)
        beta = self.get_node(elt_number, 2)
        gamma = self.get_node(elt_number, 0)

        # get position of nodes
        apos = alpha.get_position()
        bpos = beta.get_position()
        cpos = gamma.get_position()

        g_nodes = np.zeros((3, 2))
        # Transformation Function
        # x(u, v) = alpha + u(beta - alpha) + v(gamma - alpha)
        # Transform of the node (0, .5)
        g_nodes[0, 0] = (apos[0] + .5*(cpos[0] - apos[0]))
        g_nodes[0, 1] = (apos[1] + .5*(cpos[1] - apos[1]))
        # Transform of the node (.5, 0)
        g_nodes[1, 0] = (apos[0] + .5*(bpos[0] - apos[0]))
        g_nodes[1, 1] = (apos[1] + .5*(bpos[1] - apos[1]))
        # Transform of the node (.5, .5)
        g_nodes[2, 0] = (apos[0] + .5*(bpos[0] - apos[0]) + .5*(cpos[0] - apos[0]))
        g_nodes[2, 1] = (apos[1] + .5*(bpos[1] - apos[1]) + .5*(cpos[1] - apos[1]))

        return g_nodes

    def element_area(self, elt_number):
        e = self.elts[elt_number]
        n = self.nodes[e[0]]
        xbase = n.get_position()
        dx = np.zeros((2, 2))
        for ivert in [1, 2]:
            other_node_number = e[ivert]
            dx[ivert - 1, :] = self.nodes[other_node_number].get_position()
            for idir in range(2):
                dx[ivert-1, idir] -= xbase[idir]
        # WARNING: the following calculation is correct for triangles in 2D *only*.
        area = np.abs(dx[0, 0]*dx[1, 1] - dx[1, 0]*dx[0, 1])/2
        return area

    def average_element_area(self):
        area = 0
        for i in range(self.get_num_elts()):
            area += self.element_area(i)
        area /= self.get_num_elts()
        return area
        
    def gauss_quad(self, elt_number, f_values):
        # Using second order Gaussian Quadrature formula
        # 1/3*Area*[f(0, .5) + f(.5, 0) + f(.5, .5)]
        area = self.element_area(elt_number)
        integral = 1/3 * area*(np.sum(f_values))
        return integral

    def gauss_quad1d(self, f_values, boundary_vertices, e):
        # Two point Gaussian Quadrature in one dimension
        # Find length of element on boundary
        # length/2(f(-1/sqrt(3)) + f(1/sqrt(3)))
        length = self.boundary_length(boundary_vertices, e)
        integral = length/2*(f_values[0] + f_values[1])
        return integral

    def centroid(self, elt_number):
        e = self.elts[elt_number]
        retval = np.zeros(2)
        for i in range(2):
            retval[i] = 0.0
        for ivert in range(3):
            n = self.nodes[e[ivert]]
            x = n.get_position()
            for idir in range(2):
                retval[idir] += x[idir]
        for idir in range(2):
            retval[idir]/=3
        return retval

    # def interpolate_to_centroid(self, f_nodes):
    #     num_elts = self.get_num_elts()
    #     centroids = np.zeros(num_elts)
    #     for e in range(num_elts):
    #         area = self.element_area(e)
    #         for n in range(3):
    #             node = self.get_node(e, n)
    #             if node.is_interior():
    #                 id = node.get_interior_node_id()
    #                 centroids[e] += f_nodes[id]
    #     centroids /= 3
    #     return centroids

    # def nearest_neighbor(self, node_id):
    #     node = self.node(node_id)
    #     distance = 1e8
    #     neighbor = None
    #     nodex, nodey = node.get_position()
    #     for i in range(self.num_nodes):
    #         if i==node_id:
    #             continue
    #         n = self.node(i)
    #         x, y = n.get_position()
    #         norm = np.sqrt((nodex-x)**2 + (nodey-y)**2)
    #         if norm < distance:
    #             distance = norm
    #             neighbor = i
    #     return self.node(neighbor), distance
        
# def reinsert(grid, internal_solution):
#         nodes = grid.get_num_nodes()
#         full_vector = np.zeros(nodes)
#         for i in range(nodes):
#             n = grid.node(i)
#             if n.is_interior():
#                 full_vector[i] = internal_solution[n.get_interior_node_id()]
#         return full_vector
