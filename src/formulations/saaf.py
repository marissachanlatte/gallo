import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
import matplotlib.tri as tri
import itertools as itr
import sys
from numba import jit
from fe import *

class SAAF():
    def __init__(self, grid, mat_data):
        self.fegrid = grid
        self.mat_data = mat_data
        self.num_groups = self.mat_data.get_num_groups()
        self.xmax = self.fegrid.get_boundary("xmax")
        self.ymax = self.fegrid.get_boundary("ymax")
        self.xmin = self.fegrid.get_boundary("xmin")
        self.ymin = self.fegrid.get_boundary("ymin")

        # S2 hard-coded
        ang_one = .5773503
        ang_two = -.5773503
        angles = itr.product([ang_one, ang_two], repeat=2)
        self.angs = np.zeros((4, 2))
        for i, ang in enumerate(angles):
            self.angs[i] = ang
    #@jit
    def make_lhs(self, angles, group_id):
        k = self.fegrid.get_num_nodes()
        E = self.fegrid.get_num_elts()
        matrices = []
        boundary_positions = []
        sparse_matrix = sps.lil_matrix((k, k))
        for e in range(E):
            # Determine material index of element
            midx = self.fegrid.get_mat_id(e)
            # Get sigt and precomputed inverse
            inv_sigt = self.mat_data.get_inv_sigt(midx, group_id)
            sig_t = self.mat_data.get_sigt(midx, group_id)
            # Determine basis functions for element
            coef = self.fegrid.basis(e)
            # Determine Gauss Nodes for element
            g_nodes = self.fegrid.gauss_nodes(e)
            for n in range(3):
                # Get global node
                n_global = self.fegrid.get_node(e, n)
                # Get global node id
                nid = n_global.get_node_id()
                # Coefficients of basis functions b[0] + b[1]x + b[2]y
                bn = coef[:, n]
                # Array of values of basis function evaluated at interior gauss nodes
                fn_vals = np.zeros(3)
                for i in range(3):
                    fn_vals[i] = self.fegrid.evaluate_basis_function(bn, g_nodes[i])
                for ns in range(3):
                    # Get global node
                    ns_global = self.fegrid.get_node(e, ns)
                    # Get node IDs
                    nsid = ns_global.get_node_id()
                    # Coefficients of basis function
                    bns = coef[:, ns]
                    # Array of values of basis function evaluated at interior gauss nodes
                    fns_vals = np.zeros(3)
                    for i in range(3):
                        fns_vals[i] = (bns[0] + bns[1] * g_nodes[i, 0] +
                                       bns[2] * g_nodes[i, 1])
                    # Calculate gradients
                    ngrad = self.fegrid.gradient(e, n)
                    nsgrad = self.fegrid.gradient(e, ns)

                    # Multiply basis functions together at the gauss nodes
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
                    #Check if boundary nodes
                    if not n_global.is_interior() and not ns_global.is_interior():
                        # Assign boundary id, marks end of region along boundary where basis function is nonzero
                        bid = nsid
                        # Figure out what boundary you're on
                        if (nid==nsid) and (self.fegrid.is_corner(nid)):
                            # If on a corner, figure out what normal we should use
                            verts = self.fegrid.boundary_nonzero(nid, e)
                            if verts == -1: # Means the whole element is a corner
                                # We have to calculate boundary integral twice, once for each other vertex
                                # Find the other vertices
                                all_verts = np.array(self.fegrid.element(e).get_vertices())
                                vert_local_idx = np.where(all_verts == nid)[0][0]
                                other_verts = np.delete(all_verts, vert_local_idx)
                                # Calculate boundary integrals for other vertices
                                for vtx in other_verts:
                                    bid = vtx
                                    normal = self.assign_normal(nid, bid)
                                    if angles@normal > 0:
                                        xis = self.fegrid.gauss_nodes1d([nid, bid], e)
                                        boundary_integral = self.calculate_boundary_integral(nid, bid, xis, bn, bns, e)
                                        sparse_matrix[nid, nsid] += angles@normal*boundary_integral
                                continue
                            else:
                                bid = verts[1]
                        normal = self.assign_normal(nid, bid)
                        if type(normal)==int:
                            continue
                        if angles@normal > 0:
                            # Get Gauss Nodes for the element
                            xis = self.fegrid.gauss_nodes1d([nid, nsid], e)
                            boundary_integral = self.calculate_boundary_integral(nid, bid, xis, bn, bns, e)
                            sparse_matrix[nid, nsid] += angles@normal*boundary_integral
                        else:
                            pass
        return sparse_matrix

    #@jit
    def make_rhs(self, group_id, q, angles, phi_prev=None, psi_prev=None):
        angles = np.array(angles)
        # Get num elements
        E = self.fegrid.get_num_elts()
        # Get num interior nodes
        n = self.fegrid.get_num_nodes()
        rhs_at_node = np.zeros(n)
        # Interpolate Phi
        # Setup xy
        x = np.zeros(n)
        y = np.zeros(n)
        positions = (self.fegrid.node(i).get_position() for i in range(n))
        for i, pos in enumerate(positions):
            x[i], y[i] = pos
        # Setup triangles
        elts = self.fegrid.get_num_elts()
        triangles = np.array([self.fegrid.element(i).get_vertices() for i in range(elts)])
        triang = tri.Triangulation(x, y, triangles=triangles)
        # Interpolate Phis in All Groups
        G = self.num_groups
        for e in range(E):
            midx = self.fegrid.get_mat_id(e)
            inv_sigt = self.mat_data.get_inv_sigt(midx, group_id)
            sig_t = self.mat_data.get_sigt(midx, group_id)
            coef = self.fegrid.basis(e)
            # Determine basis functions for element
            coef = self.fegrid.basis(e)
            # Determine Gauss Nodes for element
            g_nodes = self.fegrid.gauss_nodes(e)
            for n in range(3):
                n_global = self.fegrid.get_node(e, n)
                # Coefficients of basis functions b[0] + b[1]x + b[2]y
                bn = coef[:, n]
                # Array of values of basis function evaluated at interior gauss nodes
                fn_vals = np.zeros(3)
                for i in range(3):
                    fn_vals[i] = self.fegrid.evaluate_basis_function(bn, g_nodes[i])
                # Get node ids
                nid = n_global.get_node_id()
                ngrad = self.fegrid.gradient(e, n)
                area = self.fegrid.element_area(e)
                # Find Phi at Gauss Nodes
                phi_vals = np.zeros((G, 3))
                for g in range(G):
                    interp = tri.LinearTriInterpolator(triang, phi_prev[g])
                    for i in range(3):
                        phi_vals[g, i] = interp(g_nodes[i, 0], g_nodes[i, 1])
                # First Scattering Term
                # Multiply Phi & Basis Function
                product = fn_vals*phi_vals
                integral_product = np.zeros(G)
                for g in range(G):
                    integral_product[g] = self.fegrid.gauss_quad(e, product[g])
                ssource = self.compute_scattering_source(midx, integral_product, group_id)
                rhs_at_node[nid] += ssource/(4*np.pi)
                # Second Scattering Term
                integral = np.zeros(G)
                for g in range(G):
                    integral[g] = self.fegrid.gauss_quad(e, phi_vals[g]*(angles@ngrad))
                ssource = self.compute_scattering_source(midx, integral, group_id)
                rhs_at_node[nid] += inv_sigt*ssource/(4*np.pi)
                # First Fixed Source Term
                q_fixed = q[e]/(4*np.pi)
                rhs_at_node[nid] += q_fixed*(area/3)
                # Second Fixed Source Term
                rhs_at_node[nid] += inv_sigt*q_fixed*(angles@ngrad)*area
        return rhs_at_node

    def compute_scattering_source(self, midx, phi, group_id):
        scatmat = self.mat_data.get_sigs(midx)
        G = self.num_groups
        s = sum(scatmat[group_id, g_prime]*phi[g_prime] for g_prime in range(G) if group_id != g_prime)
        s += scatmat[group_id, group_id]*phi[group_id]
        # for g in range(G):
        #     ss = scatmat[g_prime, group_id]
        #     if ss != 0:
        #         ssource += scatmat[group_id, group_id]*phi[g_prime]
        return s

    #@jit
    def assign_normal(self, nid, bid):
        pos_n = self.fegrid.node(nid).get_position()
        pos_ns = self.fegrid.node(bid).get_position()
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

    #@jit
    def assign_incident(self, nid, angles, psi_prev):
        pos = self.fegrid.node(nid).get_position()
        # figure out which boundary
        reflection = np.ones(2)
        if self.fegrid.is_corner(nid):
            reflection = np.array([-1, -1])
        elif pos[0] == self.xmax or pos[0] == self.xmin:
            reflection = np.array([-1, 1])
        elif pos[1] == self.ymax or pos[1] == self.ymin:
            reflection = np.array([1, -1])
        else:
            raise RuntimeError("Boundary Error")
        incident_angle = angles*reflection
        ia_idx = np.where((self.angs == incident_angle).all(axis=1))[0][0]
        incident_flux = psi_prev[ia_idx, nid]
        return incident_flux

    #@jit
    def calculate_boundary_integral(self, nid, bid, xis, bn, bns, e):
        pos_n = self.fegrid.node(nid).get_position()
        pos_ns = self.fegrid.node(bid).get_position()
        gauss_nodes = np.zeros((2, 2))
        if (pos_n[0] == self.xmax and pos_ns[0] == self.xmax):
            gauss_nodes[0] = [self.xmax, xis[0]]
            gauss_nodes[1] = [self.xmax, xis[1]]
        elif (pos_n[0] == self.xmin and pos_ns[0] == self.xmin):
            gauss_nodes[0] = [self.xmin, xis[0]]
            gauss_nodes[1] = [self.xmin, xis[1]]
        elif (pos_n[1] == self.ymax and pos_ns[1] == self.ymax):
            gauss_nodes[0] = [xis[0], self.ymax]
            gauss_nodes[1] = [xis[1], self.ymax]
        elif (pos_n[1] == self.ymin and pos_ns[1] == self.ymin):
            gauss_nodes[0] = [xis[0], self.ymin]
            gauss_nodes[1] = [xis[1], self.ymin]
        else:
            boundary_integral = 0
            return boundary_integral
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
        boundary_integral = self.fegrid.gauss_quad1d(g_vals, [nid, bid], e)
        return boundary_integral

    #@jit
    def get_scalar_flux(self, group_id, source, phi_prev, psi_prev=None):
        # TODO: S4 Angular Quadrature for 2D
        #S2 quadrature
        ang_one = .5773503
        ang_two = -.5773503
        angles = itr.product([ang_one, ang_two], repeat=2)
        scalar_flux = 0
        N = self.fegrid.get_num_nodes()
        ang_fluxes = np.zeros((4, N))
        # Iterate over all angle possibilities
        for i, ang in enumerate(angles):
            ang = np.array(ang)
            ang_fluxes[i] = self.get_ang_flux(group_id, source, ang, phi_prev)
            # Multiplying by weight and summing for quadrature
            scalar_flux += np.pi*ang_fluxes[i]
        return scalar_flux, ang_fluxes

    #@jit
    def get_ang_flux(self, group_id, source, ang, phi_prev, psi_prev=None):
        lhs = self.make_lhs(ang, group_id)
        rhs = self.make_rhs(group_id, source, ang, phi_prev, psi_prev)
        ang_flux = linalg.cg(lhs, rhs)[0]
        return ang_flux

    # def build_scattering_matrix(self):
    #     k = self.fegrid.get_num_nodes()
    #     E = self.fegrid.get_num_elts()
    #     G = self.num_groups
    #     scattering_matrix = np.zeros((G, G, k, k))
    #     for g in range(G):
    #         for g_prime in range(G):
    #             for e in range(E):
    #                 midx = self.fegrid.get_mat_id(e)
    #                 scatmat = self.mat_data.get_sigs(midx)
    #                 sig_s = scatmat[g, g_prime]
    #                 coef = self.fegrid.basis(e)
    #                 g_nodes = self.fegrid.gauss_nodes(e)
    #                 for n in range(3):
    #                     n_global = self.fegrid.get_node(e, n)
    #                     nid = n_global.get_node_id()
    #                     bn = coef[:, n]
    #                     fn_vals = np.zeros(3)
    #                     for i in range(3):
    #                         fn_vals[i] = self.fegrid.evaluate_basis_function(bn, g_nodes[i])
    #                     for ns in range(3):
    #                         ns_global = self.fegrid.get_node(e, ns)
    #                         nsid = ns_global.get_node_id()
    #                         bns = coef[:, ns]
    #                         fns_vals = np.zeros(3)
    #                         for i in range(3):
    #                             fns_vals[i] = self.fegrid.evaluate_basis_function(bns, g_nodes[i])
    #                         f_vals = np.zeros(3)
    #                         for i in range(3):
    #                            f_vals[i] = fn_vals[i] * fns_vals[i]
    #                         integral = self.fegrid.gauss_quad(e, f_vals)
    #                         scattering_matrix[g, g_prime, nid, nsid] += sig_s*integral
    #     return scattering_matrix
    #
    # def make_external_source(self, q):
    #     E = self.fegrid.get_num_elts()
    #     n = self.fegrid.get_num_nodes()
    #     external_source = np.zeros(n)
    #     for e in range(E):
    #         for n in range(3):
    #             n_global = self.fegrid.get_node(e, n)
    #             nid = n_global.get_node_id()
    #             ngrad = self.fegrid.gradient(e, n)
    #             area = self.fegrid.element_area(e)
    #             q_fixed = q[e]/(4*np.pi)
    #             external_source[nid] += q_fixed*(area/3)
    #     return external_source

    def solve_in_group(self, source, group_id, phi_prev, max_iter=50, tol=1e-2):
        print("Starting Group ", group_id)
        E = self.fegrid.get_num_elts()
        N = self.fegrid.get_num_nodes()
        for i in range(max_iter):
            print("Within-Group Iteration: ", i)
            phi, ang_fluxes = self.get_scalar_flux(group_id, source, phi_prev)
            norm = np.linalg.norm(phi-phi_prev[group_id], 2)
            print("Norm: ", norm)
            if norm < tol:
                break
            phi_prev[group_id] = np.copy(phi)
        if i==max_iter:
            print("Warning: maximum number of iterations reached in solver")
        print("Finished Group ", group_id)
        print("Number of Within-Group Iterations: ", i+1)
        print("Final Phi Norm: ", norm)
        return phi, ang_fluxes

    def gauss_seidel(self, A, b, phi, tol):
        m, n = np.shape(A)
        for it_count in range(1000):
            phi_prev = np.copy(phi)
            for i in range(m):
                s = sum(A[i, j]*phi[j] for j in range(n) if i != j)
                phi[i] = (b[i] - s)/A[i, i]
            if np.allclose(phi, phi_prev, rtol=tol):
                break
        return phi

    def solve_outer(self, source, max_iter=50, tol=1e-2):
        G = self.num_groups
        N = self.fegrid.get_num_nodes()
        phis = np.zeros((G, N))
        ang_fluxes = np.zeros((G, 4, N))
        for it_count in range(1000):
            print("Gauss-Seidel Iteration: ", it_count)
            phis_prev = np.copy(phis)
            for g in range(G):
                p, a = self.solve_in_group(source, g, phis)
                phis[g] = p
                ang_fluxes[g] = a
            res = np.max(np.abs(phis_prev - phis))
            print("GS Norm: ", res)
            if np.allclose(phis, phis_prev, rtol=tol):
                break
        return phis, ang_fluxes

    # def solve_outer(self, source, max_iter=50, tol=1e-2):
    #     G = self.num_groups
    #     N = self.fegrid.get_num_nodes()
    #     phis = np.zeros((G, N))
    #     ang_fluxes = np.zeros((G, 4, N))
    #     it = 0
    #     res = 100
    #     for it_count in range(1000):
    #         print("Gauss-Seidel Iteration: ", it_count)
    #         phis_prev = np.copy(phis)
    #         for g in range(G):
    #             p, a = self.solve_in_group(source, g, phis)
    #             phis[g] = p
    #             ang_fluxes[g] = a
    #             # GS Update
    #             s = sum(np.matmul(H[g, g_prime], phis[g_prime]) for g_prime in range(G) if g != g_prime)
    #             H_inv = np.linalg.inv(H[g, g])
    #             phis[g] = np.matmul(H_inv, (q - s))
    #         res = np.max(np.abs(phis_prev - phis))
    #         print("GS Norm: ", res)
    #         if np.allclose(phis, phis_prev, rtol=tol):
    #             break
    #     return phis, ang_fluxes
