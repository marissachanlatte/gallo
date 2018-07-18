import itertools as itr
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
from fe import *

class NDA():
    def __init__(self, grid, mat_data):
        self.fegrid = grid
        self.mat_data = mat_data
        self.num_groups = self.mat_data.get_num_groups()
        self.num_nodes = self.fegrid.num_nodes
        self.num_elts = self.fegrid.num_elts
        self.num_gnodes = self.fegrid.num_gauss_nodes
        self.num_angs = self.fegrid.num_angs
        self.angs = self.fegrid.angs
        self.weights = self.fegrid.weights

    def make_lhs(self, group_id, ho_sols):
        sparse_matrix = sps.lil_matrix((self.num_nodes, self.num_nodes))
        # Solve higher order equation
        if ho_sols !=0:
            phi = np.array([ho_sols[0]])
            psi = np.array([ho_sols[1]])
        # Interpolate Phi
        triang = self.fegrid.setup_triangulation()
        for e in range(self.num_elts):
            # Determine material index of element
            midx = self.fegrid.get_mat_id(e)
            # Get Diffusion coefficient for material
            D = self.mat_data.get_diff(midx, group_id)
            # Get removal cross section
            sig_r = self.mat_data.get_sigr(midx, group_id)
            inv_sigt = self.mat_data.get_inv_sigt(midx, group_id)
            # Determine basis functions for element
            coef = self.fegrid.basis(e)
            # Determine Gauss Nodes for element
            g_nodes = self.fegrid.gauss_nodes(e)
            # Dertmine Gradients for Each Basis function
            grad = np.array([self.fegrid.gradient(e, i) for i in range(3)])
            vertices = self.fegrid.element(e).vertices
            if ho_sols !=0:
                # Find Phi at Gauss Nodes
                phi_vals = self.fegrid.phi_at_gauss_nodes(triang, phi, g_nodes)
                # Find Psi at Gauss Nodes
                psi_vals = np.array([self.fegrid.phi_at_gauss_nodes(triang, psi[:, i], g_nodes) for i in range(self.num_angs)])
                # Find Psi at vertices
                psi_at_verts = np.zeros((self.num_angs, 3))
                for m in range(self.num_angs):
                    for i, vtx in enumerate(vertices):
                        psi_at_verts[m, i] = psi[0, m, vtx]
            for n in range(3):
                # Get global node
                n_global = self.fegrid.get_node(e, n)
                # Get node IDs
                nid = n_global.id
                # Coefficients of basis functions b[0] + b[1]x + b[2]y
                bn = coef[:, n]
                # Array of values of basis function evaluated at gauss nodes
                fn_vals = np.array([
                    self.fegrid.evaluate_basis_function(bn, g_nodes[i])
                    for i in range(self.num_gnodes)])
                for ns in range(3):
                    # Get global node
                    ns_global = self.fegrid.get_node(e, ns)
                    nsid = ns_global.id
                    # Coefficients of basis function
                    bns = coef[:, ns]
                    # Array of values of basis function evaluated at gauss nodes
                    fns_vals = np.array([
                        self.fegrid.evaluate_basis_function(bns, g_nodes[i])
                        for i in range(self.num_gnodes)])

                    # Integrate for A (basis function derivatives)
                    area = self.fegrid.element_area(e)
                    inprod = np.dot(grad[n], grad[ns])
                    sparse_matrix[nid, nsid] += D * area * inprod

                    # Integrate for B (basis functions multiplied)
                    basis_integral = self.fegrid.gauss_quad(e, fn_vals * fns_vals)
                    sparse_matrix[nid, nsid] += sig_r * basis_integral

                    # Calculate drift_vector
                    if ho_sols == 0:
                        drift_vector = np.zeros((self.num_gnodes, 2))
                    else:
                        drift_vector = self.compute_drift_vector(inv_sigt, D, grad, phi_vals[0], psi_at_verts)
                        #drift_vector = self.compute_drift_vector(inv_sigt, D, grad[n], phi_vals[0], psi_vals[:, 0])

                    # Integrate drift_vector@gradient*basis_function
                    drift_product = np.array([drift_vector[i]*fn_vals[i] for i in range(self.num_gnodes)])
                    integral = self.fegrid.gauss_quad(e, drift_product@grad[ns])
                    sparse_matrix[nid, nsid] += integral
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
                                all_verts = np.array(self.fegrid.element(e).get_vertices())
                                vert_local_idx = np.where(all_verts == nid)[0][0]
                                other_verts = np.delete(all_verts, vert_local_idx)
                                # Calculate boundary integrals for other vertices
                                for vtx in other_verts:
                                    bid = vtx
                                    if ho_sols != 0:
                                        normal = self.fegrid.assign_normal(nid, bid)
                                        xis = self.fegrid.gauss_nodes1d([nid, bid], e)
                                        phi_bd = self.fegrid.phi_at_gauss_nodes(triang, phi, xis)
                                        psi_bd = np.array([self.fegrid.phi_at_gauss_nodes(triang, psi[:, i], xis) for i in range(self.num_angs)])
                                        basis_product = self.fegrid.boundary_basis_product(nid, bid, xis, bn, bns, e)
                                        kappa = self.compute_kappa(normal, phi_bd[0], psi_bd[:, 0])
                                        boundary_integral = self.fegrid.gauss_quad1d(kappa*basis_product, [nid, bid], e)
                                        sparse_matrix[nid,nsid] += boundary_integral
                                continue
                            else:
                                bid = verts[1]
                        normal = self.fegrid.assign_normal(nid, bid)
                        if isinstance(normal, int):
                            continue
                        if ho_sols != 0:
                            # Get Gauss Nodes for the element
                            xis = self.fegrid.gauss_nodes1d([nid, bid], e)
                            phi_bd = self.fegrid.phi_at_gauss_nodes(triang, phi, xis)
                            psi_bd = np.array([self.fegrid.phi_at_gauss_nodes(triang, psi[:, i], xis) for i in range(self.num_angs)])
                            basis_product = self.fegrid.boundary_basis_product(nid, bid, xis, bn, bns, e)
                            kappa = self.compute_kappa(normal, phi_bd[0], psi_bd[:, 0])
                            boundary_integral = self.fegrid.gauss_quad1d(kappa*basis_product, [nid, bid], e)
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
            for n in range(3):
                n_global = self.fegrid.get_node(e, n)
                # Coefficients of basis functions b[0] + b[1]x + b[2]y
                bn = coef[:, n]
                # Array of values of basis function evaluated at interior gauss nodes
                fn_vals = np.array([self.fegrid.evaluate_basis_function(bn, g_nodes[i]) for i in range(self.num_gnodes)])
                # Get node ids
                nid = n_global.id
                area = self.fegrid.element_area(e)
                # Find Phi at Gauss Nodes
                phi_vals = self.fegrid.phi_at_gauss_nodes(triang, phi_prev, g_nodes)
                ssource = self.compute_scattering_source(midx, phi_vals, group_id)
                # Multiply Scattering & Basis Function
                product = fn_vals * ssource
                integral_product = self.fegrid.gauss_quad(e, product)
                rhs_at_node[nid] += integral_product
                rhs_at_node[nid] += area*source[group_id, e]*1/3
        return rhs_at_node

    def compute_scattering_source(self, midx, phi, group_id):
        scatmat = self.mat_data.get_sigs(midx)
        ssource = np.zeros(self.num_gnodes)
        for gnode in range(self.num_gnodes):
            for g_prime in range(self.num_groups):
                if g_prime != group_id:
                    ssource[gnode] += scatmat[g_prime, group_id]*phi[g_prime, gnode]
        return ssource

    def compute_kappa(self, normal, phi, psi):
        kappa = np.zeros(3)
        # Use interpolated version of kappa
        for node in range(3):
            for i, ang in enumerate(self.angs):
                kappa[node] += self.weights[i]*np.abs(ang@normal)*psi[i, node]
            kappa[node] /= phi[node]
        return kappa

    def compute_drift_vector(self, inv_sigt, D, grad, phi, psi):
        # Calculate drift_vector
        drift_vector = np.zeros((self.num_gnodes, 2))
        for node in range(self.num_gnodes):
            for m, ang in enumerate(self.angs):
                deriv = psi[m]@grad
                dir_deriv = psi[m]@(grad@ang)
                drift_vector[node] += self.weights[m]*inv_sigt*ang*dir_deriv
                drift_vector[node] -= self.weights[m]*D*deriv
                # drift_vector[node] += self.weights[i]*inv_sigt*ang*(ang@grad)*psi[i, node]
                # drift_vector[node] -= self.weights[i]*D*grad*psi[i, node]
            drift_vector[node] /= phi[node]
        return drift_vector
