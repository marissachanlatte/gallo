import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
from fe import *

class Diffusion():
    def __init__(self, grid, mat_data):
        self.fegrid = grid
        self.mat_data = mat_data
        self.num_groups = self.mat_data.get_num_groups()
        self.num_nodes = self.fegrid.get_num_nodes()
        self.num_elts = self.fegrid.get_num_elts()

    def make_lhs(self, group_id, ho_sols=None, boundary=None):
        E = self.fegrid.get_num_elts()
        sparse_matrix = sps.lil_matrix((self.num_nodes, self.num_nodes))
        for e in range(E):
            # Determine material index of element
            midx = self.fegrid.get_mat_id(e)
            # Get Diffusion coefficient for material
            D = self.mat_data.get_diff(midx, group_id)
            # Get removal cross section
            sig_r = self.mat_data.get_sigr(midx, group_id)
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
                    nid = n_global.get_node_id()
                    nsid = ns_global.get_node_id()
                    # Calculate gradients
                    ngrad = self.fegrid.gradient(e, n)
                    nsgrad = self.fegrid.gradient(e, ns)

                    # Integrate for A (basis function derivatives)
                    area = self.fegrid.element_area(e)
                    inprod = np.dot(ngrad, nsgrad)
                    A = D * area * inprod

                    # Multiply basis functions together
                    f_vals = np.zeros(3)
                    for i in range(3):
                        f_vals[i] = fn_vals[i] * fns_vals[i]

                    # Integrate for B (basis functions multiplied)
                    integral = self.fegrid.gauss_quad(e, f_vals)
                    C = sig_r * integral

                    sparse_matrix[nid, nsid] += A + C
                    if boundary=="reflecting": continue
                    if not n_global.is_interior() and not ns_global.is_interior():
                        # Assign boundary id, marks end of region along
                        # boundary where basis function is nonzero
                        bid = nsid
                        # Figure out what boundary you're on
                        if (nid == nsid) and (self.fegrid.is_corner(nid)):
                            # If on a corner, figure out what normal we should use
                            verts = self.fegrid.boundary_nonzero(nid, e)
                            if verts == -1:  # Means the whole element is a corner
                                # We have to calculate boundary integral twice,
                                # once for each other vertex
                                # Find the other vertices
                                all_verts = np.array(self.fegrid.element(e).get_vertices())
                                vert_local_idx = np.where(all_verts == nid)[0][0]
                                other_verts = np.delete(all_verts, vert_local_idx)
                                # Calculate boundary integrals for other vertices
                                for vtx in other_verts:
                                    bid = vtx
                                    xis = self.fegrid.gauss_nodes1d([nid, bid], e)
                                    boundary_integral = self.fegrid.calculate_boundary_integral(nid, bid, xis, bn, bns, e)
                                    sparse_matrix[nid,nsid] += boundary_integral
                                continue
                            else:
                                bid = verts[1]
                        # Get Gauss Nodes for the element
                        xis = self.fegrid.gauss_nodes1d([nid, nsid], e)
                        boundary_integral = self.fegrid.calculate_boundary_integral(nid, bid, xis, bn, bns, e)
                        sparse_matrix[nid, nsid] += boundary_integral
        return sparse_matrix

    def make_rhs(self, group_id, source, phi_prev, eigenvalue=False):
        rhs_at_node = np.zeros(self.num_nodes)
        # Interpolate Phi
        triang = self.fegrid.setup_triangulation()
        for e in range(self.num_elts):
            midx = self.fegrid.get_mat_id(e)
            # Determine basis functions for element
            coef = self.fegrid.basis(e)
            # Determine Gauss Nodes for element
            g_nodes = self.fegrid.gauss_nodes(e)
            if eigenvalue:
                nu = self.mat_data.get_nu(midx, group_id)
                sig_f = self.mat_data.get_sigf(midx, group_id)
            for n in range(3):
                n_global = self.fegrid.get_node(e, n)
                # Check if boundary node
                if not n_global.is_interior():
                    continue
                # Coefficients of basis functions b[0] + b[1]x + b[2]y
                bn = coef[:, n]
                # Array of values of basis function evaluated at interior gauss nodes
                fn_vals = np.zeros(3)
                for i in range(3):
                    fn_vals[i] = self.fegrid.evaluate_basis_function(
                        bn, g_nodes[i])
                # Get node ids
                nid = n_global.get_node_id()
                area = self.fegrid.element_area(e)
                # Find Phi at Gauss Nodes
                phi_vals = self.fegrid.phi_at_gauss_nodes(triang, phi_prev, g_nodes)
                # Multiply Phi & Basis Function
                product = fn_vals * phi_vals
                integral_product = np.zeros(self.num_groups)
                for g in range(self.num_groups):
                    integral_product[g] = self.fegrid.gauss_quad(e, product[g])
                ssource = self.compute_scattering_source(
                    midx, integral_product, group_id)
                rhs_at_node[nid] += ssource
                if eigenvalue:
                    rhs_at_node[nid] += nu*sig_f*integral_product[group_id]
                else:
                    rhs_at_node[nid] += area*source[group_id, e]*1/3
        return rhs_at_node

    def make_fission_source(self, group_id, phi_prev):
        fission_source = np.zeros(self.num_nodes)
        triang = self.fegrid.setup_triangulation()
        for e in range(self.num_elts):
            midx = self.fegrid.get_mat_id(e)
            chi = self.mat_data.get_chi(midx, group_id)
            sigf = np.array([self.mat_data.get_sigf(midx, g_prime) for g_prime in range(self.num_groups)])
            nu = np.array([self.mat_data.get_nu(midx, g_prime) for g_prime in range(self.num_groups)])
            # Determine basis functions for element
            coef = self.fegrid.basis(e)
            # Determine Gauss Nodes for element
            g_nodes = self.fegrid.gauss_nodes(e)
            for n in range(3):
                n_global = self.fegrid.get_node(e, n)
                nid = n_global.get_node_id()
                # Check if boundary node
                if not n_global.is_interior():
                    continue
                # Coefficients of basis functions b[0] + b[1]x + b[2]y
                bn = coef[:, n]
                # Array of values of basis function evaluated at interior gauss nodes
                fn_vals = np.array([self.fegrid.evaluate_basis_function(bn, g_nodes[i]) for i in range(3)])
                # Find Phi at Gauss Nodes
                phi_vals = self.fegrid.phi_at_gauss_nodes(triang, phi_prev, g_nodes)
                # Multiply Phi & Basis Function
                product = fn_vals * phi_vals
                integral_product = np.array([self.fegrid.gauss_quad(e, product[g])
                                             for g in range(self.num_groups)])
                fiss = chi*np.sum(np.array([nu[g_prime]*sigf[g_prime]*integral_product[g_prime]
                    for g_prime in range(self.num_groups)]))
                #fission_source[nid] += fiss
                interp = tri.LinearTriInterpolator(triang, phi_prev[group_id])
                centroid = self.fegrid.centroid(e)
                phi_centroid = interp(centroid[0], centroid[1])
                area = self.fegrid.element_area(e)
                fission_source[nid] += nu[group_id]*sigf[group_id]*phi_centroid*area/3
        return fission_source

    def compute_scattering_source(self, midx, phi, group_id):
        scatmat = self.mat_data.get_sigs(midx)
        ssource = 0
        for g_prime in range(self.num_groups):
            if g_prime != group_id:
                ssource += scatmat[g_prime, group_id]*phi[g_prime]
        return ssource

    def compute_collapsed_fission(self, phi):
        triang = self.fegrid.setup_triangulation()
        collapsed_fission = 0
        for g in range(self.num_groups):
            for e in range(self.num_elts):
                midx = self.fegrid.get_mat_id(e)
                nu = self.mat_data.get_nu(midx, g)
                sig_f = self.mat_data.get_sigf(midx, g)
                area = self.fegrid.element_area(e)
                g_nodes = self.fegrid.gauss_nodes(e)
                phi_vals = self.fegrid.phi_at_gauss_nodes(triang, phi, g_nodes)
                phi_integral = self.fegrid.gauss_quad(e, phi_vals[g])
                collapsed_fission += nu*sig_f*phi_integral
        return collapsed_fission

    ### OLD SOLVER FROM CS294 PROJECT. NOT CURRENTLY HOOKED UP TO ANYTHING ####
    # def solve(self, lhs, rhs, problem_type, group_id, max_iter=1000, tol=1e-5):
    #     if problem_type=="fixed_source":
    #         internal_nodes = linalg.cg(lhs, rhs)[0]
    #         return internal_nodes
    #     elif problem_type=="eigenvalue":
    #         E = self.fegrid.get_num_elts()
    #         N = self.fegrid.get_num_interior_nodes()
    #         phi = np.zeros(N)
    #         phi_prev = np.ones(N)
    #         k_prev = np.sum(phi_prev)
    #         # renormalize
    #         phi_prev /= k_prev
    #
    #         for i in range(max_iter):
    #             # setup rhs
    #             phi_centroid = self.fegrid.interpolate_to_centroid(phi_prev)
    #             f_centroids = self.make_eigen_source(group_id, phi_centroid)
    #             # Integrate
    #             rhs = self.make_rhs(f_centroids)
    #             # solve
    #             phi = linalg.cg(lhs, rhs)[0]
    #             # compute k by integrating phi
    #             phi_centroids = self.fegrid.interpolate_to_centroid(phi)
    #             integral = self.make_rhs(phi_centroids)
    #             k = np.sum(integral)
    #             # renormalize
    #             phi /= k
    #             norm = np.linalg.norm(phi-phi_prev, 2)
    #             knorm = np.abs(k - k_prev)
    #             if knorm < tol and norm < tol:
    #                 break
    #             phi_prev = phi
    #             k_prev = k
    #             print("Eigenvalue Iteration: ", i)
    #             print("Norm: ", norm)
    #
    #         if i==max_iter:
    #             print("Warning: maximum number of iterations reached in eigenvalue solver")
    #
    #         max = np.max(phi)
    #         phi /= max
    #
    #         print("Number of Iterations: ", i)
    #         print("Final Phi Norm: ", norm)
    #         print("Final k Norm: ", knorm)
    #         return phi, k
    #
    #     else:
    #         print("Problem type must be fixed_source or eigenvalue")
