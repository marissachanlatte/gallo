import itertools as itr

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
import matplotlib.tri as tri

class SAAF():
    def __init__(self, grid, mat_data):
        self.fegrid = grid
        self.mat_data = mat_data
        self.num_groups = self.mat_data.get_num_groups()
        self.xmax = self.fegrid.xmax
        self.ymax = self.fegrid.ymax
        self.xmin = self.fegrid.xmin
        self.ymin = self.fegrid.ymin
        self.num_nodes = self.fegrid.num_nodes
        self.num_elts = self.fegrid.num_elts
        self.num_gnodes = self.fegrid.num_gauss_nodes

    def make_lhs(self, angles, group_id):
        sparse_matrix = sps.lil_matrix((self.num_nodes, self.num_nodes))
        for e in range(self.num_elts):
            elt = self.fegrid.element(e)
            # Determine material index of element
            midx = elt.mat_id
            # Get sigt and precomputed inverse
            inv_sigt = self.mat_data.get_inv_sigt(midx, group_id)
            sig_t = self.mat_data.get_sigt(midx, group_id)
            # Determine basis functions for element
            coef = self.fegrid.basis(e)
            # Determine Gauss Nodes for element
            g_nodes = self.fegrid.gauss_nodes(e)
            for n in range(3):
                # Get global node
                n_global = self.fegrid.node(e, n)
                # Get global node id
                nid = n_global.id
                # Coefficients of basis functions b[0] + b[1]x + b[2]y
                bn = coef[:, n]
                # Array of values of basis function evaluated at gauss nodes
                fn_vals = np.array([self.fegrid.evaluate_basis_function(bn, g_nodes[i])
                    for i in range(self.num_gnodes)])
                for ns in range(3):
                    # Get global node
                    ns_global = self.fegrid.node(e, ns)
                    # Get node IDs
                    nsid = ns_global.id
                    # Coefficients of basis function
                    bns = coef[:, ns]
                    # Array of values of basis function evaluated at gauss nodes
                    fns_vals = np.array([self.fegrid.evaluate_basis_function(bns, g_nodes[i])
                        for i in range(self.num_gnodes)])
                    # Calculate gradients
                    grad = np.array([self.fegrid.gradient(e, i) for i in [n, ns]])
                    # Multiply basis functions together at the gauss nodes
                    f_vals = fn_vals * fns_vals
                    # Integrate for A (basis function derivatives)
                    area = self.fegrid.element_area(e)
                    A = inv_sigt * (angles @ grad[0]) * (angles @ grad[1]) * area
                    # Integrate for B (basis functions multiplied)
                    integral = self.fegrid.gauss_quad(e, f_vals)
                    C = sig_t * integral
                    sparse_matrix[nid, nsid] += A + C
                    # Check if boundary nodes
                    if not n_global.is_interior and not ns_global.is_interior:
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
                                all_verts = np.array(self.fegrid.element(e).vertices)
                                vert_local_idx = np.where(all_verts == nid)[0][0]
                                other_verts = np.delete(all_verts, vert_local_idx)
                                # Calculate boundary integrals for other vertices
                                for vtx in other_verts:
                                    bid = vtx
                                    normal = self.fegrid.assign_normal(nid, bid)
                                    if angles @ normal > 0:
                                        xis = self.fegrid.gauss_nodes1d(
                                            [nid, bid], e)
                                        basis_product = self.fegrid.boundary_basis_product(nid, bid, xis, bn, bns, e)
                                        boundary_integral = self.fegrid.gauss_quad1d(basis_product, [nid, bid], e)
                                        sparse_matrix[nid, nsid] += angles @ normal * boundary_integral
                                continue
                            else:
                                bid = verts[1]
                        normal = self.fegrid.assign_normal(nid, bid)
                        if isinstance(normal, int):
                            continue
                        if angles @ normal > 0:
                            # Get Gauss Nodes for the element
                            xis = self.fegrid.gauss_nodes1d([nid, bid], e)
                            basis_product = self.fegrid.boundary_basis_product(nid, bid, xis, bn, bns, e)
                            boundary_integral = self.fegrid.gauss_quad1d(basis_product, [nid, bid], e)
                            sparse_matrix[nid, nsid] += angles @ normal * boundary_integral
                        else:
                            pass
        return sparse_matrix

    def make_rhs(self, group_id, source, angles, angle_id, phi_prev=None):
        angles = np.array(angles)
        rhs_at_node = np.zeros(self.num_nodes)
        # Interpolate Phi
        triang = self.fegrid.setup_triangulation()
        for e in range(self.num_elts):
            elt = self.fegrid.element(e)
            midx = elt.mat_id
            inv_sigt = self.mat_data.get_inv_sigt(midx, group_id)
            # Determine basis functions for element
            coef = self.fegrid.basis(e)
            # Determine Gauss Nodes for element
            g_nodes = self.fegrid.gauss_nodes(e)
            for n in range(3):
                n_global = self.fegrid.node(e, n)
                # Coefficients of basis functions b[0] + b[1]x + b[2]y
                bn = coef[:, n]
                # Array of values of basis function evaluated at interior gauss nodes
                fn_vals = np.array([self.fegrid.evaluate_basis_function(bn, g_nodes[i])
                    for i in range(self.num_gnodes)])
                # Get node ids
                nid = n_global.id
                ngrad = self.fegrid.gradient(e, n)
                area = self.fegrid.element_area(e)
                # Find Phi at Gauss Nodes
                phi_vals = self.fegrid.phi_at_gauss_nodes(triang, phi_prev, g_nodes)
                # First Scattering Term
                # Multiply Phi & Basis Function
                product = fn_vals * phi_vals
                integral_product = np.zeros(self.num_groups)
                for g in range(self.num_groups):
                    integral_product[g] = self.fegrid.gauss_quad(e, product[g])
                ssource = self.compute_scattering_source(
                    midx, integral_product, group_id)
                rhs_at_node[nid] += ssource / (4 * np.pi)
                # Second Scattering Term
                integral = np.zeros(self.num_groups)
                for g in range(self.num_groups):
                    integral[g] = self.fegrid.gauss_quad(e, phi_vals[g]*(angles@ngrad))
                ssource = self.compute_scattering_source(
                    midx, integral, group_id)
                rhs_at_node[nid] += inv_sigt*ssource/(4*np.pi)
                q_fixed = source[group_id, e] / (4 * np.pi)
                rhs_at_node[nid] += q_fixed * (area / 3)
                # Second Fixed Source Term
                rhs_at_node[nid] += inv_sigt*q_fixed*(angles@ngrad)*area
        return rhs_at_node

    def compute_scattering_source(self, midx, phi, group_id):
        scatmat = self.mat_data.get_sigs(midx)
        ssource = 0
        for g_prime in range(self.num_groups):
            ssource += scatmat[g_prime, group_id]*phi[g_prime]
        return ssource
