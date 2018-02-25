import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
import itertools as itr
from fe import *

class SAAF():
    def __init__(self, grid, mat_data):
        self.fegrid = grid
        self.mat_data = mat_data
        self.num_groups = self.mat_data.get_num_groups()
        self.matrices = []

        
    def make_lhs(self, angles):
        #k = self.fegrid.get_num_interior_nodes()
        k = self.fegrid.get_num_nodes()
        E = self.fegrid.get_num_elts()
        matrices = []
        for g in range(self.num_groups):
            sparse_matrix = sps.lil_matrix((k, k))
            for e in range(E):
                # Determine material index of element
                midx = self.fegrid.get_mat_id(e)
                # Get sigt and precomputed inverse
                inv_sigt = self.mat_data.get_inv_sigt(midx, g)
                sig_t = self.mat_data.get_sigt(midx, g)
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
                        #nid = n_global.get_interior_node_id()
                        #nsid = ns_global.get_interior_node_id()
                        nid = n_global.get_node_id()
                        nsid = ns_global.get_node_id()
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
                            # TODO: Figure out Omega
                            area = self.fegrid.element_area(e)
                            A = inv_sigt*area*(angles@ngrad)*(angles@nsgrad)

                            # Integrate for B (basis functions multiplied)
                            integral = self.fegrid.gauss_quad(e, f_vals)
                            C = sig_t * integral

                            sparse_matrix[nid, nsid] += A + C
            matrices.append(sparse_matrix)
        return matrices

    def make_rhs(self, group_id, q, angles, phi_prev=None):
        # Get num elements
        E = self.fegrid.get_num_elts()
        # Get num interior nodes
        #n = self.fegrid.get_num_interior_nodes()
        n = self.fegrid.get_num_nodes()
        rhs_at_node = np.zeros(n)
        for e in range(E):
            midx = self.fegrid.get_mat_id(e)
            inv_sigt = self.mat_data.get_inv_sigt(midx, group_id)
            sig_s = self.mat_data.get_sigs(midx, group_id)
            for n in range(3):
                n_global = self.fegrid.get_node(e, n)
                # Check if boundary node
                if not n_global.is_interior():
                    continue
                # Get node ids
                #nid = n_global.get_interior_node_id()
                nid = n_global.get_node_id()
                area = self.fegrid.element_area(e)
                ngrad = self.fegrid.gradient(e, n)
                rhs_at_node[nid] += ((sig_s*phi_prev[nid] + q[e]) 
                     + (angles*(inv_sigt*(sig_s*phi_prev[nid] + q[e])))@ngrad)*1/3*area
        return rhs_at_node

    def get_matrix(self, group_id):
        if group_id == "all":
            return self.matrices
        else:
            return self.matrices[group_id]

    def get_scalar_flux(self, group_id, source, phi_prev):
        # TODO: S4 Angular Quadrature for 2D
        # ang_one = 0.3500212
        # ang_two = 0.8688903
        ang_one = .5773503
        ang_two = -.5773503
        angles = itr.product([ang_one, ang_two], repeat=2)
        scalar_flux = 0
        ang_fluxes = []
        # Iterate over all angle possibilities
        for ang in angles:
            ang = np.array(ang)
            ang_flux = self.get_ang_flux(group_id, source, ang, phi_prev)
            ang_fluxes.append(ang_flux)
        ### REFLECTING BOUNDARY CONDITIONS ###

            # Multiplying by weight and summing for quadrature
            scalar_flux += ang_flux
        return scalar_flux, ang_fluxes

    def get_ang_flux(self, group_id, source, ang, phi_prev):
        lhs = self.make_lhs(ang)[0]
        rhs = self.make_rhs(group_id, source, ang, phi_prev)
        ang_flux = linalg.cg(lhs, rhs)[0]
        return ang_flux

    def solve(self, source, problem_type, group_id, max_iter=1000, tol=1e-4):
        E = self.fegrid.get_num_elts()
        #N = self.fegrid.get_num_interior_nodes()
        N = self.fegrid.get_num_nodes()
        phi = np.zeros(N)
        phi_prev = np.zeros(N)
        for i in range(max_iter):
            phi, ang_fluxes = self.get_scalar_flux(0, source, phi_prev)
            norm = np.linalg.norm(phi-phi_prev, 2)
            if norm < tol:
                break
            phi_prev = phi
            print("Iteration: ", i)
            print("Norm: ", norm)

        if i==max_iter:
            print("Warning: maximum number of iterations reached in solver")

        print("Number of Iterations: ", i)
        print("Final Phi Norm: ", norm)
        return phi, ang_fluxes
















