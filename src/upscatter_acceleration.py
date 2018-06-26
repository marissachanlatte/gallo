import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg

class UA():
    def __init__(self, operator, phis, phis_prev, ho_sols):
        self.op = operator
        self.phis = phis
        self.phis_prev = phis_prev
        self.ho_sols = ho_sols
        self.fegrid = self.op.fegrid
        self.mat_data = self.op.mat_data
        self.num_nodes = self.fegrid.get_num_nodes()
        self.num_groups = self.mat_data.get_num_groups()
        self.num_elts = self.fegrid.get_num_elts()

    def calculate_correction(self):
        lhs = self.correction_lhs()
        rhs = self.correction_rhs()
        correction = linalg.cg(lhs, rhs)[0]
        return correction

    def correction_lhs(self):
        sparse_matrix = sps.lil_matrix((self.num_nodes, self.num_nodes))
        ho_phi = self.ho_sols[0]
        ho_psi = self.ho_sols[1]
        for e in range(self.num_elts):
            midx = self.fegrid.get_mat_id(e)
            diffs = np.array([self.mat_data.get_diff(midx, g) for g in range(self.num_groups)])
            D = np.sum(diffs)
            sig_a = np.sum(np.array([self.mat_data.get_siga(midx, g) for g in range(self.num_groups)]))
            inv_sigt = np.array([self.mat_data.get_inv_sigt(midx, g) for g in range(self.num_groups)])
            # Determine basis functions for element
            coef = self.fegrid.basis(e)
            # Determine Gauss Nodes for element
            g_nodes = self.fegrid.gauss_nodes(e)
            for n in range(3):
                # Coefficients of basis functions b[0] + b[1]x + b[2]y
                bn = coef[:, n]
                # Array of values of basis function evaluated at gauss nodes
                fn_vals = np.array([bn[0] + bn[1] * g_nodes[i, 0] + bn[2] * g_nodes[i, 1] for i in range(3)])
                n_global = self.fegrid.get_node(e, n)
                for ns in range(3):
                    # Coefficients of basis function
                    bns = coef[:, ns]
                    # Array of values of basis function evaluated at gauss nodes
                    fns_vals = np.array([bns[0] + bns[1] * g_nodes[i, 0] + bns[2] * g_nodes[i, 1] for i in range(3)])
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
                    C = sig_a * integral

                    # Calculate drift_vector
                    drift_vector = np.zeros(2)
                    for g in range(self.num_groups):
                        drift_vector += self.op.compute_drift_vector(inv_sigt[g], diffs[g], ngrad, ho_phi[g], ho_psi[g], nid)

                    # Integrate drift_vector@gradient*basis_function
                    integral = self.fegrid.gauss_quad(e, (drift_vector@ngrad)*fn_vals)
                    E = integral

                    sparse_matrix[nid, nsid] += A + C + E
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

    def correction_rhs(self):
        rhs_at_node = np.zeros(self.num_nodes)
        triang = self.fegrid.setup_triangulation()
        for e in range(self.num_elts):
            midx = self.fegrid.get_mat_id(e)
            coef = self.fegrid.basis(e)
            g_nodes = self.fegrid.gauss_nodes(e)
            for n in range(3):
                n_global = self.fegrid.get_node(e, n)
                if not n_global.is_interior():
                    continue
                # Coefficients of basis functions b[0] + b[1]x + b[2]y
                bn = coef[:, n]
                # Array of values of basis function evaluated at interior gauss nodes
                fn_vals = np.array([self.fegrid.evaluate_basis_function(bn, g_nodes[i]) for i in range(3)])
                # Get node ids
                nid = n_global.get_node_id()

                # Add Phi Prevs
                # Find Phi at Gauss Nodes
                phi_vals = self.fegrid.phi_at_gauss_nodes(triang, self.phis_prev, g_nodes)
                # Multiply Phi & Basis Function
                product = fn_vals * phi_vals
                integral = np.array([self.fegrid.gauss_quad(e, product[g]) for g in range(self.num_groups)])
                ssource = np.sum(np.array([self.partial_scat(midx, integral, g) for g in range(self.num_groups)]))
                rhs_at_node[nid] += ssource

                # Subtract Phi Prevs
                # Find Phi at Gauss Nodes
                phi_vals = self.fegrid.phi_at_gauss_nodes(triang, self.phis, g_nodes)
                # Multiply Phi & Basis Function
                product = fn_vals * phi_vals
                integral = np.array([self.fegrid.gauss_quad(e, product[g]) for g in range(self.num_groups)])
                ssource = np.sum(np.array([self.partial_scat(midx, integral, g) for g in range(self.num_groups)]))
                rhs_at_node[nid] -= ssource

        return rhs_at_node

    def partial_scat(self, midx, phi, group_id):
        scatmat = self.mat_data.get_sigs(midx)
        ssource = 0
        for g_prime in range(group_id+1, self.num_groups):
            if g_prime != group_id:
                ssource += scatmat[g_prime, group_id]*phi[g_prime]
        return ssource
