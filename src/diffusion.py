import numpy as np
import scipy.sparse as sps


class DiffusionOperator():
    def __init__(self, grid, mat_data):
        k = grid.get_num_interior_nodes()
        self.matrix = sps.csc_matrix((k, k))
        self.fegrid = grid
        self.mat_data = mat_data

        # Get num elements
        E = grid.get_num_elts()

        for e in range(E):
            # Determine material index of element
            midx = self.fegrid.get_mat_id(e)
            # Get Diffusion coefficient for material
            D = self.mat_data.get_diff(midx)
            # Get absorption cross section for material
            sig_a = self.mat_data.get_siga(midx)
            # Determine basis functions for element
            C = grid.basis(e)
            # Determine Gauss Nodes for element
            g_nodes = grid.gauss_nodes(e)
            for n in range(3):
                # Coefficients of basis functions b[0] + b[1]x + b[2]y
                bn = C[:, n]
                # Array of values of basis function evaluated at gauss nodes
                fn_vals = np.zeros(3)
                for i in range(3):
                    fn_vals[i] = (
                        bn[0] + bn[1] * g_nodes[i, 0] + bn[2] * g_nodes[i, 1])
                # Get global node
                n_global = grid.get_node(e, n)

                for ns in range(3):
                    # Coefficients of basis function
                    bns = C[:, ns]
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

                        self.matrix[nid, nsid] += A + C
