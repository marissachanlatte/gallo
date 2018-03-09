import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
import itertools as itr
import sys
from fe import *

class SAAF():
    def __init__(self, grid, mat_data):
        self.fegrid = grid
        self.mat_data = mat_data
        self.num_groups = self.mat_data.get_num_groups()
        #self.matrices = []
        self.xmax = self.fegrid.get_boundary("xmax")
        self.ymax = self.fegrid.get_boundary("ymax")
        self.xmin = self.fegrid.get_boundary("xmin")
        self.ymin = self.fegrid.get_boundary("ymin")
     
    def make_lhs(self, angles):
        k = self.fegrid.get_num_nodes()
        E = self.fegrid.get_num_elts()
        matrices = []
        boundary_positions = []
        for g in range(self.num_groups):
            sparse_matrix = sps.lil_matrix((k, k))
            for e in range(E):
                # Determine material index of element
                midx = self.fegrid.get_mat_id(e)
                # Get sigt and precomputed inverse
                inv_sigt = self.mat_data.get_inv_sigt(midx, g)
                sig_t = self.mat_data.get_sigt(midx, g)
                sig_s = self.mat_data.get_sigs(midx, g)
                # Determine basis functions for element
                coef = self.fegrid.basis(e)
                # Determine Gauss Nodes for element
                g_nodes = self.fegrid.gauss_nodes(e)
                for n in range(3):
                    # Get global node
                    n_global = self.fegrid.get_node(e, n)
                    # Coefficients of basis functions b[0] + b[1]x + b[2]y
                    bn = coef[:, n]
                    if n_global.is_interior():
                        # Array of values of basis function evaluated at interior gauss nodes
                        fn_vals = np.zeros(3)
                        for i in range(3):
                            fn_vals[i] = self.fegrid.evaluate_basis_function(bn, g_nodes[i])
                            #fn_vals[i] = (bn[0] + bn[1] * g_nodes[i, 0] + bn[2] * g_nodes[i, 1])
                    for ns in range(3):
                        # Get global node
                        ns_global = self.fegrid.get_node(e, ns)
                        # Get node IDs
                        nid = n_global.get_node_id()
                        nsid = ns_global.get_node_id()
                        # Coefficients of basis function
                        bns = coef[:, ns]
                        # Check if boundary nodes
                        if not n_global.is_interior() and not ns_global.is_interior():         
                            # Get Gauss Nodes for the element
                            xis = self.fegrid.gauss_nodes1d([nid, nsid], e)
                            gauss_nodes = np.zeros((2, 2))
                            # Figure out what boundary you're on
                            pos_n = np.array(n_global.get_position())
                            pos_ns = np.array(ns_global.get_position())
                            if (pos_n[0] == self.xmax and pos_ns[0] == self.xmax):
                                normal = np.array([1, 0])
                                gauss_nodes[0] = [self.xmax, xis[0]]
                                gauss_nodes[1] = [self.xmax, xis[1]]
                            elif (pos_n[0] == self.xmin and pos_ns[0] == self.xmin):
                                normal = np.array([-1, 0])
                                gauss_nodes[0] = [self.xmin, xis[0]]
                                gauss_nodes[1] = [self.xmin, xis[1]]
                            elif (pos_n[1] == self.ymax and pos_ns[1] == self.ymax):
                                normal = np.array([0, 1])
                                gauss_nodes[0] = [xis[0], self.ymax]
                                gauss_nodes[1] = [xis[1], self.ymax]
                            elif (pos_n[1] == self.ymin and pos_ns[1] == self.ymin):
                                normal = np.array([0, -1])
                                gauss_nodes[0] = [xis[0], self.ymin]
                                gauss_nodes[1] = [xis[1], self.ymin]
                            else:
                                continue
                            # Value of first basis function at boundary gauss nodes
                            gn_vals = np.zeros(2)
                            gn_vals[0] = self.fegrid.evaluate_basis_function(bn, gauss_nodes[0])
                            gn_vals[1] = self.fegrid.evaluate_basis_function(bn, gauss_nodes[1])
                            # Values of second basis function at boundary gauss nodes
                            gns_vals = np.zeros(2)
                            gns_vals[0] = self.fegrid.evaluate_basis_function(bns, gauss_nodes[0])
                            gns_vals[1] = self.fegrid.evaluate_basis_function(bns, gauss_nodes[1])
                            # Multiply basis functions together
                            g_vals = gn_vals*gns_vals
                            # Integrate over length of element on boundary
                            boundary_integral = self.fegrid.gauss_quad1d(g_vals, [nid, nsid], e)
                            if angles@normal > 0:
                                sparse_matrix[nid, nsid] += angles@normal*boundary_integral
                            else:
                                pass
                        elif not ns_global.is_interior() or not n_global.is_interior():
                            pass
                        else:
                            # Array of values of basis function evaluated at interior gauss nodes
                            fns_vals = np.zeros(3)
                            for i in range(3):
                                fns_vals[i] = (bns[0] + bns[1] * g_nodes[i, 0] +
                                               bns[2] * g_nodes[i, 1])
                            # Calculate gradients
                            ngrad = self.fegrid.gradient(e, n)
                            nsgrad = self.fegrid.gradient(e, ns)

                            # Multiply basis functions together
                            f_vals = np.zeros(3)
                            for i in range(3):
                               f_vals[i] = fn_vals[i] * fns_vals[i]

                            # Integrate for A (basis function derivatives)
                            area = self.fegrid.element_area(e)
                            A = inv_sigt*(angles@ngrad)*(angles@nsgrad)*area

                            # Integrate for B (basis functions multiplied)
                            integral = self.fegrid.gauss_quad(e, f_vals)
                            C = sig_t * integral

                            sparse_matrix[nid, nsid] += A + C
            matrices.append(sparse_matrix)
        return matrices

    def make_rhs(self, group_id, q, angles, boundary, phi_prev=None):
        angles = np.array(angles)
        # Get num elements
        E = self.fegrid.get_num_elts()
        # Get num interior nodes
        n = self.fegrid.get_num_nodes()
        rhs_at_node = np.zeros(n)
        for e in range(E):
            midx = self.fegrid.get_mat_id(e)
            inv_sigt = self.mat_data.get_inv_sigt(midx, group_id)
            sig_t = self.mat_data.get_sigt(midx, group_id)
            sig_s = self.mat_data.get_sigs(midx, group_id)
            coef = self.fegrid.basis(e)
            for n in range(3):
                n_global = self.fegrid.get_node(e, n)
                # Get node ids
                nid = n_global.get_node_id()
                ngrad = self.fegrid.gradient(e, n)
                # Check if boundary node
                if not n_global.is_interior():
                    if boundary == "vacuum":
                        continue
                    elif boundary=="reflecting":
                        # Figure out what boundary you're one
                        pos_n = np.array(n_global.get_position()) 
                        if (pos_n[0] == self.xmax):
                            normal = np.array([1, 0])
                            reflection = np.array([-1, 1])
                        elif (pos_n[0] == self.xmin):
                            normal = np.array([-1, 0])
                            reflection = np.array([-1, 1])
                        elif (pos_n[1] == self.ymax):
                            normal = np.array([0, 1])
                            reflection = np.array([1, -1])
                        elif (pos_n[1] == self.ymin):
                            normal = np.array([0, -1])
                            reflection = np.array([1, -1])
                        else:
                            raise RuntimeError("Boundary error.")
                        if angles@normal > 0:
                            angle_reflected = reflection*angles
                            area = self.fegrid.element_area(e)
                            Q = sig_s*phi_prev[nid] + q[e]/(4*np.pi)
                            rhs_at_node[nid] += Q*area/3 + inv_sigt*Q*(angle_reflected@ngrad)*area
                    else:
                        raise RuntimeError("Boundary condition not implemented.")
                else:
                    area = self.fegrid.element_area(e)
                    Q = sig_s*phi_prev[nid] + q[e]/(4*np.pi)
                    rhs_at_node[nid] += Q*area/3 + inv_sigt*Q*(angles@ngrad)*area
        return rhs_at_node

    def get_scalar_flux(self, group_id, source, phi_prev, boundary):
        # TODO: S4 Angular Quadrature for 2D
        #S2 quadrature
        ang_one = .5773503
        ang_two = -.5773503
        angles = itr.product([ang_one, ang_two], repeat=2)
        scalar_flux = 0
        ang_fluxes = []
        # Iterate over all angle possibilities
        for ang in angles:
            ang = np.array(ang)
            ang_flux = self.get_ang_flux(group_id, source, ang, phi_prev, boundary)
            ang_fluxes.append(ang_flux)
            # Multiplying by weight and summing for quadrature
            scalar_flux += np.pi*ang_flux
        return scalar_flux, ang_fluxes

    def get_ang_flux(self, group_id, source, ang, phi_prev, boundary):
        lhs = self.make_lhs(ang)[0]
        rhs = self.make_rhs(group_id, source, ang, boundary, phi_prev)
        ang_flux = linalg.cg(lhs, rhs)[0]
        return ang_flux

    def solve(self, source, problem_type, group_id, boundary, max_iter=1000, tol=1e-4):
        E = self.fegrid.get_num_elts()
        N = self.fegrid.get_num_nodes()
        phi = np.zeros(N)
        phi_prev = np.zeros(N)
        for i in range(max_iter):
            phi, ang_fluxes = self.get_scalar_flux(0, source, phi_prev, boundary)
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
















