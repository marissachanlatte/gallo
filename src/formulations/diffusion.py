import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
from fe import *

class Diffusion():
    def __init__(self, grid, mat_data):
        self.fegrid = grid
        self.mat_data = mat_data
        self.num_groups = self.mat_data.get_num_groups()
        self.matrices = self.make_lhs()
        
    def make_lhs(self):
        k = self.fegrid.get_num_interior_nodes()
        E = self.fegrid.get_num_elts()
        matrices = []
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
                coef = self.fegrid.basis(e)
                # Determine Gauss Nodes for element
                g_nodes = self.fegrid.gauss_nodes(e)
                for n in range(3):
                    # Coefficients of basis functions b[0] + b[1]x + b[2]y
                    bn = coef[:, n]
                    # Array of values of basis function evaluated at gauss nodes
                    fn_vals = np.zeros(3)
                    for i in range(3):
                        fn_vals[i] = (
                            bn[0] + bn[1] * g_nodes[i, 0] + bn[2] * g_nodes[i, 1])
                    # Get global node
                    n_global = self.fegrid.get_node(e, n)
                    for ns in range(3):
                        # Coefficients of basis function
                        bns = coef[:, ns]
                        # Array of values of basis function evaluated at gauss nodes
                        fns_vals = np.zeros(3)
                        for i in range(3):
                            fns_vals[i] = (bns[0] + bns[1] * g_nodes[i, 0] +
                                           bns[2] * g_nodes[i, 1])
                        # Get global node
                        ns_global = self.fegrid.get_node(e, ns)
                        # Get node IDs
                        nid = n_global.get_interior_node_id()
                        nsid = ns_global.get_interior_node_id()
                        # Check if boundary nodes
                        if not ns_global.is_interior() or not n_global.is_interior():
                            continue
                        else:
                            # Calculate gradients
                            ngrad = self.fegrid.gradient(e, n)
                            nsgrad = self.fegrid.gradient(e, ns)

                            # Multiply basis functions together
                            f_vals = np.zeros(3)
                            for i in range(3):
                                f_vals[i] = fn_vals[i] * fns_vals[i]

                            # Integrate for A (basis function derivatives)
                            area = self.fegrid.element_area(e)
                            inprod = np.dot(ngrad, nsgrad)
                            A = D * area * inprod

                            # Integrate for B (basis functions multiplied)
                            integral = self.fegrid.gauss_quad(e, f_vals)
                            C = sig_a * integral

                            sparse_matrix[nid, nsid] += A + C
            matrices.append(sparse_matrix)
        return matrices

    def make_rhs(self, f_centroids):
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
        return rhs_at_node

    def make_eigen_source(self, group_id, phi_prev):
        E = self.fegrid.get_num_elts()
        f_centroids = np.zeros(E)
        for e in range(E):
            midx = self.fegrid.get_mat_id(e)
            nu = self.mat_data.get_nu(midx, group_id)
            sig_f = self.mat_data.get_sigf(midx, group_id)
            f_centroids[e] = nu*sig_f*phi_prev[e]
        return f_centroids

    def get_matrix(self, group_id):
        if group_id == "all":
            return self.matrices
        else:
            return self.matrices[group_id]

    def solve(self, lhs, rhs, problem_type, group_id, max_iter=1000, tol=1e-5):
        if problem_type=="fixed_source":
            internal_nodes = linalg.cg(lhs, rhs)[0]
            return internal_nodes
        elif problem_type=="eigenvalue":
            E = self.fegrid.get_num_elts()
            N = self.fegrid.get_num_interior_nodes()
            phi = np.zeros(N)
            phi_prev = np.ones(N)
            k_prev = np.sum(phi_prev)
            # renormalize
            phi_prev /= k_prev

            for i in range(max_iter):
                # setup rhs
                phi_centroid = self.fegrid.interpolate_to_centroid(phi_prev)
                f_centroids = self.make_eigen_source(group_id, phi_centroid)
                # Integrate
                rhs = self.make_rhs(f_centroids)
                # solve
                phi = linalg.cg(lhs, rhs)[0]
                # compute k by integrating phi
                phi_centroids = self.fegrid.interpolate_to_centroid(phi)
                integral = self.make_rhs(phi_centroids)
                k = np.sum(integral)
                # renormalize
                phi /= k
                norm = np.linalg.norm(phi-phi_prev, 2)
                knorm = np.abs(k - k_prev)
                if knorm < tol and norm < tol:
                    break
                phi_prev = phi
                k_prev = k
                print("Eigenvalue Iteration: ", i)
                print("Norm: ", norm)

            if i==max_iter:
                print("Warning: maximum number of iterations reached in eigenvalue solver")

            max = np.max(phi)
            phi /= max

            print("Number of Iterations: ", i)
            print("Final Phi Norm: ", norm)
            print("Final k Norm: ", knorm)
            return phi, k

        else:
            print("Problem type must be fixed_source or eigenvalue")















