import numpy as np
import scipy.sparse as sps


class Diffusion():
    def __init__(self, grid, mat_data):
        k = grid.get_num_interior_nodes()
        E = grid.get_num_elts()
        self.matrices = []
        self.fegrid = grid
        self.mat_data = mat_data
        self.num_groups = self.mat_data.get_num_groups()
        
        for g in range(self.num_groups):
            sparse_matrix = sps.lil_matrix((k, k))
            for e in range(E):
                # Determine material index of element
                midx = self.fegrid.get_mat_id(e)
                # Get Diffusion coefficient for material
                D = self.mat_data.get_diff(midx, g)
                # Get absorption cross section for material
                sig_a = self.mat_data.get_siga(midx, g)
                # Determine basis functions for element
                coef = grid.basis(e)
                # Determine Gauss Nodes for element
                g_nodes = grid.gauss_nodes(e)
                for n in range(3):
                    # Coefficients of basis functions b[0] + b[1]x + b[2]y
                    bn = coef[:, n]
                    # Array of values of basis function evaluated at gauss nodes
                    fn_vals = np.zeros(3)
                    for i in range(3):
                        fn_vals[i] = (
                            bn[0] + bn[1] * g_nodes[i, 0] + bn[2] * g_nodes[i, 1])
                    # Get global node
                    n_global = grid.get_node(e, n)
                    for ns in range(3):
                        # Coefficients of basis function
                        bns = coef[:, ns]
                        # Array of values of basis function evaluated at gauss nodes
                        fns_vals = np.zeros(3)
                        for i in range(3):
                            fns_vals[i] = (bns[0] + bns[1] * g_nodes[i, 0] +
                                           bns[2] * g_nodes[i, 1])
                        # Get global node
                        ns_global = grid.get_node(e, ns)
                        # Get node IDs
                        nid = n_global.get_interior_node_id()
                        nsid = ns_global.get_interior_node_id()
                        # Check if boundary nodes
                        if not ns_global.is_interior() or not n_global.is_interior():
                            continue
                        else:
                            # Calculate gradients
                            ngrad = grid.gradient(e, n)
                            nsgrad = grid.gradient(e, ns)

                            # Multiply basis functions together
                            f_vals = np.zeros(3)
                            for i in range(3):
                                f_vals[i] = fn_vals[i] * fns_vals[i]

                            # Integrate for A (basis function derivatives)
                            area = grid.element_area(e)
                            inprod = np.dot(ngrad, nsgrad)
                            A = D * area * inprod

                            # Integrate for B (basis functions multiplied)
                            integral = grid.gauss_quad(e, f_vals)
                            C = sig_a * integral

                            sparse_matrix[nid, nsid] += A + C
            self.matrices.append(sparse_matrix)

    def make_rhs(self, f_centroids):
        rhses = []
        for g in range(self.num_groups):
            # Get num interior nodes
            n = self.fegrid.get_num_interior_nodes()
            rhs_at_node = np.zeros(n)

            # Get num elements
            E = self.fegrid.get_num_elts()

            for e in range(E):
                for n in range(3):
                    n_global = self.fegrid.get_node(e, n)
                    # Check if boundary node
                    if not n_global.is_interior():
                        continue
                    # Get node ids
                    nid = n_global.get_interior_node_id()
                    area = self.fegrid.element_area(e)
                    rhs_at_node[nid] += area*f_centroids[e]*1/3
            rhses.append(rhs_at_node)
        return rhses

    def get_matrix(self, group_id):
        if group_id == "all":
            return self.matrices
        else:
            return self.matrices[group_id]













