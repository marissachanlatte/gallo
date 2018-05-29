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
        self.xmax = self.fegrid.get_boundary("xmax")
        self.ymax = self.fegrid.get_boundary("ymax")
        self.xmin = self.fegrid.get_boundary("xmin")
        self.ymin = self.fegrid.get_boundary("ymin")
        self.num_nodes = self.fegrid.get_num_nodes()
        self.elts = self.fegrid.get_num_elts()

        # S2 hard-coded
        ang_one = .5773503
        ang_two = -.5773503
        angles = itr.product([ang_one, ang_two], repeat=2)
        self.angs = np.zeros((4, 2))
        for i, ang in enumerate(angles):
            self.angs[i] = ang

    def make_lhs(self, angles, group_id):
        sparse_matrix = sps.lil_matrix((self.num_nodes, self.num_nodes))
        for e in range(self.elts):
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
                # Array of values of basis function evaluated at gauss nodes
                fn_vals = np.array([
                    self.fegrid.evaluate_basis_function(bn, g_nodes[i])
                    for i in range(3)
                ])
                for ns in range(3):
                    # Get global node
                    ns_global = self.fegrid.get_node(e, ns)
                    # Get node IDs
                    nsid = ns_global.get_node_id()
                    # Coefficients of basis function
                    bns = coef[:, ns]
                    # Array of values of basis function evaluated at gauss nodes
                    fns_vals = np.array([
                        self.fegrid.evaluate_basis_function(bns, g_nodes[i])
                        for i in range(3)
                    ])
                    # Calculate gradients
                    grad = np.array(
                        [self.fegrid.gradient(e, i) for i in [n, ns]])
                    # Multiply basis functions together at the gauss nodes
                    f_vals = fn_vals * fns_vals
                    # Integrate for A (basis function derivatives)
                    area = self.fegrid.element_area(e)
                    A = inv_sigt * (angles @ grad[0]) * (
                        angles @ grad[1]) * area
                    # Integrate for B (basis functions multiplied)
                    integral = self.fegrid.gauss_quad(e, f_vals)
                    C = sig_t * integral
                    sparse_matrix[nid, nsid] += A + C
                    # Check if boundary nodes
                    if not n_global.is_interior(
                    ) and not ns_global.is_interior():
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
                                all_verts = np.array(
                                    self.fegrid.element(e).get_vertices())
                                vert_local_idx = np.where(
                                    all_verts == nid)[0][0]
                                other_verts = np.delete(
                                    all_verts, vert_local_idx)
                                # Calculate boundary integrals for other vertices
                                for vtx in other_verts:
                                    bid = vtx
                                    normal = self.assign_normal(nid, bid)
                                    if angles @ normal > 0:
                                        xis = self.fegrid.gauss_nodes1d(
                                            [nid, bid], e)
                                        boundary_integral = self.calculate_boundary_integral(
                                            nid, bid, xis, bn, bns, e)
                                        sparse_matrix[
                                            nid,
                                            nsid] += angles @ normal * boundary_integral
                                continue
                            else:
                                bid = verts[1]
                        normal = self.assign_normal(nid, bid)
                        if isinstance(normal, int):
                            continue
                        if angles @ normal > 0:
                            # Get Gauss Nodes for the element
                            xis = self.fegrid.gauss_nodes1d([nid, nsid], e)
                            boundary_integral = self.calculate_boundary_integral(
                                nid, bid, xis, bn, bns, e)
                            sparse_matrix[
                                nid,
                                nsid] += angles @ normal * boundary_integral
                        else:
                            pass
        return sparse_matrix

    def make_rhs(self, group_id, source, angles, phi_prev=None, eigenvalue=False):
        angles = np.array(angles)
        rhs_at_node = np.zeros(self.num_nodes)
        # Interpolate Phi
        triang = self.setup_triangulation()
        for e in range(self.elts):
            midx = self.fegrid.get_mat_id(e)
            inv_sigt = self.mat_data.get_inv_sigt(midx, group_id)
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
                    fn_vals[i] = self.fegrid.evaluate_basis_function(
                        bn, g_nodes[i])
                # Get node ids
                nid = n_global.get_node_id()
                ngrad = self.fegrid.gradient(e, n)
                area = self.fegrid.element_area(e)
                # Find Phi at Gauss Nodes
                phi_vals = np.zeros((self.num_groups, 3))
                for g in range(self.num_groups):
                    interp = tri.LinearTriInterpolator(triang, phi_prev[g])
                    for i in range(3):
                        phi_vals[g, i] = interp(g_nodes[i, 0], g_nodes[i, 1])
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
                    integral[g] = self.fegrid.gauss_quad(
                        e, phi_vals[g]*(angles@ngrad))
                ssource = self.compute_scattering_source(
                    midx, integral, group_id)
                rhs_at_node[nid] += inv_sigt*ssource/(4*np.pi)
                if eigenvalue:
                    rhs_at_node[nid] += source[group_id, e]*(area/3)
                else:
                    # First Fixed Source Term
                    q_fixed = source[group_id, e] / (4 * np.pi)
                    rhs_at_node[nid] += q_fixed * (area / 3)
                    # Second Fixed Source Term
                    rhs_at_node[nid] += inv_sigt*q_fixed*(angles@ngrad)*area
        return rhs_at_node

    def setup_triangulation(self):
        x = np.zeros(self.num_nodes)
        y = np.zeros(self.num_nodes)
        positions = (self.fegrid.node(i).get_position() for i in range(self.num_nodes))
        for i, pos in enumerate(positions):
            x[i], y[i] = pos
        # Setup triangles
        triangles = np.array(
            [self.fegrid.element(i).get_vertices() for i in range(self.elts)])
        triang = tri.Triangulation(x, y, triangles=triangles)
        return triang

    def make_fission_source(self, group_id, angles, phi_prev):
        angles = np.array(angles)
        E = self.fegrid.get_num_elts()
        fission_source = np.zeros(self.num_nodes)
        triang = self.setup_triangulation()
        # Interpolate Phi
        G = self.num_groups
        for e in range(E):
            midx = self.fegrid.get_mat_id(e)
            inv_sigt = self.mat_data.get_inv_sigt(midx, group_id)
            chi = self.mat_data.get_chi(midx, group_id)
            sigf = self.mat_data.get_sigf(midx, group_id)
            nu = self.mat_data.get_nu(midx, group_id)
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
                    fn_vals[i] = self.fegrid.evaluate_basis_function(
                        bn, g_nodes[i])
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
                # Multiply Phi & Basis Function
                product = fn_vals * phi_vals
                integral_product = np.zeros(G)
                for g in range(G):
                    integral_product[g] = self.fegrid.gauss_quad(e, product[g])
                integral = np.zeros(G)
                for g in range(G):
                    integral[g] = self.fegrid.gauss_quad(e, phi_vals[g]*(angles@ngrad))
                # First Fission Source Term
                fission_source[nid] += chi*nu*sigf*integral_product[group_id]/(4*np.pi)
                # Second Fission Source Term
                fission_source[nid] += inv_sigt*chi*nu*sigf*integral[group_id]/(4*np.pi)
        return fission_source

    def compute_scattering_source(self, midx, phi, group_id):
        scatmat = self.mat_data.get_sigs(midx)
        G = self.num_groups
        s = sum(scatmat[group_id, g_prime] * phi[g_prime]
                for g_prime in range(G) if group_id != g_prime)
        s += scatmat[group_id, group_id] * phi[group_id]
        return s

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
        g_vals = gn_vals * gns_vals
        # Integrate over length of element on boundary
        boundary_integral = self.fegrid.gauss_quad1d(g_vals, [nid, bid], e)
        return boundary_integral

    def get_ang_flux(self, group_id, source, ang, phi_prev, eig_bool):
        lhs = self.make_lhs(ang, group_id)
        rhs = self.make_rhs(group_id, source, ang, phi_prev, eigenvalue=eig_bool)
        ang_flux = linalg.cg(lhs, rhs)[0]
        return ang_flux

    def get_scalar_flux(self, group_id, source, phi_prev, eig_bool):
        scalar_flux = 0
        ang_fluxes = np.zeros((4, self.num_nodes))
        # Iterate over all angle possibilities
        for i, ang in enumerate(self.angs):
            ang = np.array(ang)
            ang_fluxes[i] = self.get_ang_flux(group_id, source, ang, phi_prev, eig_bool)
            scalar_flux += np.pi * ang_fluxes[i]
        return scalar_flux, ang_fluxes

    def solve_in_group(self, source, group_id, phi_prev, eig_bool, max_iter=50,
                       tol=1e-2):
        num_mats = self.mat_data.get_num_mats()
        for mat in range(num_mats):
            scatmat = self.mat_data.get_sigs(mat)
            if np.count_nonzero(scatmat) == 0:
                scattering = False
            else:
                scattering = True
        if self.num_groups > 1:
            print("Starting Group ", group_id)
        for i in range(max_iter):
            if scattering:
                print("Within-Group Iteration: ", i)
            phi, ang_fluxes = self.get_scalar_flux(group_id, source, phi_prev, eig_bool)
            if not scattering:
                break
            norm = np.linalg.norm(phi - phi_prev[group_id], 2)
            print("Norm: ", norm)
            if norm < tol:
                break
            phi_prev[group_id] = np.copy(phi)
        if i == max_iter:
            print("Warning: maximum number of iterations reached in solver")
        if self.num_groups > 1:
            print("Finished Group ", group_id)
        if scattering:
            print("Number of Within-Group Iterations: ", i + 1)
            print("Final Phi Norm: ", norm)
        return phi, ang_fluxes

    def solve_outer(self, source, eig_bool, max_iter=50, tol=1e-2):
        phis = np.ones((self.num_groups, self.num_nodes))
        ang_fluxes = np.zeros((self.num_groups, 4, self.num_nodes))
        for it_count in range(max_iter):
            if self.num_groups != 1:
                print("Gauss-Seidel Iteration: ", it_count)
            phis_prev = np.copy(phis)
            for g in range(self.num_groups):
                p, a = self.solve_in_group(source, g, phis, eig_bool)
                phis[g] = p
                ang_fluxes[g] = a
            if self.num_groups == 1:
                break
            else:
                res = np.max(np.abs(phis_prev - phis))
                print("GS Norm: ", res)
            if res < tol:
                break
        return phis, ang_fluxes

    def power_iteration(self, source, max_iter=50, tol=1e-2):
        eig_vecs = np.ones((self.num_groups, self.num_nodes))
        ang_fluxes = np.zeros((self.num_groups, 4, self.num_nodes))
        eigenvalues = np.zeros(self.num_groups)
        # initialize fission Source
        fission_source = np.ones((self.num_groups, 4, self.num_nodes))
        for it_count in range(max_iter):
            print("Eigenvalue Iteration: ", it_count)
            vec_products, ang_fluxes = self.solve_outer(fission_source, True)
            new_eig_vecs = np.array([vec_products[i]/np.linalg.norm(vec_products[i], ord=2)
                                 for i in range(G)])
            #new_eigenvalues = np.array([np.matmul(new_vecs[i].transpose(), phis[i])
            #                            for i in range(G)])
            res = np.max(np.abs(new_eig_vecs - eig_vecs))
            print("Norm: ", res)
            if res < tol:
                break
            # Calculate new fission Source
            for g in range(self.num_groups):
                for i, ang in enumerate(self.angs):
                    fission_source[g, i] = self.make_fission_source(g, ang, vec_products)
            eig_vecs = new_eig_vecs
        eigenvalues = np.array([np.matmul(eig_vecs[i].transpose(), vec_products[i]) for i in range(G)])
        print("Eigenvalues are: ", eigenvalues)
        return vec_products, ang_fluxes, eigenvalues

    def solve(self, source, eigenvalue=False):
        if eigenvalue:
            phis, ang_fluxes, eigenvalues = self.power_iteration(source)
            return phis, ang_fluxes, eigenvalues
        else:
            eig_bool = False
            phis, ang_fluxes = self.solve_outer(source, eig_bool)
            return phis, ang_fluxes
