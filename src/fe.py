import os
import numpy as np

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
    def __init__(self, position, node_id, interior_node_id, is_interior):
        self.position = position
        self.id = node_id
        self.is_interior = is_interior
        self.interior_node_id = interior_node_id

    def get_position(self):
        return self.position

    def get_node_id(self):
        return self.id

    def get_interior_node_id(self):
        return self.interior_node_id

    def is_interior(self):
        return self.is_interior

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
            self.num_interior_nodes = 0
            for i in range(self.num_nodes):
                line = nf.readline()
                #remove extra whitespace
                line = ' '.join(line.split())
                data = line.split(" ")
                node_id, x, y, boundary = data
                x = float(x)
                y = float(y) 
                is_interior = not int(boundary)
                interior_node_id = -1
                if is_interior: 
                    interior_node_id = self.num_interior_nodes
                    self.num_interior_nodes += 1
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

    def get_mat_id(self, elt_number):
        return self.element(elt_number).get_mat_id()
        
    def gradient(self, elt_number, node_number):
        # WARNING: The following only works for 2D triangular elements
        e = self.elts[elt_number].get_vertices()
        n = self.nodes[node_number].get_position()
        assert(n.is_interior())
        dx = np.zeros(2, 2)
        for ivert in range(2):
            other_node_number = e[(node_number + ivert + 1)%3]
            dx[ivert] = self.nodes[other_node_number].get_position()
            for idir in range(2):
                dx[ivert, idir] -= n[idir]

        det = dx[0, 0]*dx[1, 1] - dx[1, 0]*dx[0, 1]
        retval = np.zeros(2)
        retval[0] = (-(dx[1, 1] - dx[0, 1])/det)
        retval[1] = (-(dx[1, 0] - dx[0, 0])/det)
        return retval

    def basis(self, elt_number):
        V = np.zeros(3, 3)
        for i in range(2):
            V[i, 0] = 1
            V[i, 1] = self.get_node(elt_number, i).get_position()[0]
            V[i, 2] = self.get_node(elt_number, i).get_position()[1]
        C = np.linalg.inv(V)
        return C

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

    def gauss_quad(self, elt_number, f_values):
        # Using second order Gaussian Quadrature formula
        # 1/3*Area*[f(0, .5) + f(.5, 0) + f(.5, .5)]
        area = self.element_area(elt_number)
        integral = 1/3 * area*(np.sum(f_values))
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

