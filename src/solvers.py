import itertools as itr

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
import matplotlib.tri as tri

from formulations.diffusion import Diffusion

class Solver():
    def __init__(self, operator):
        self.op = operator
        self.mat_data = self.op.mat_data
        self.num_groups = self.op.num_groups
        self.num_nodes = self.op.num_nodes
        if not isinstance(self.op, Diffusion):
            self.angs = self.op.angs

    def get_ang_flux(self, group_id, source, ang, angle_id, phi_prev, eig_bool):
        lhs = self.op.make_lhs(ang, group_id)
        rhs = self.op.make_rhs(group_id, source, ang, angle_id, phi_prev, eigenvalue=eig_bool)
        ang_flux = linalg.cg(lhs, rhs)[0]
        return ang_flux

    def get_scalar_flux(self, group_id, source, phi_prev, eig_bool):
        scalar_flux = 0
        if isinstance(self.op, Diffusion):
            lhs = self.op.make_lhs(group_id)
            rhs = self.op.make_rhs(group_id, source, phi_prev, eigenvalue=eig_bool)
            scalar_flux = linalg.cg(lhs, rhs)[0]
            return scalar_flux
        else:
            ang_fluxes = np.zeros((4, self.num_nodes))
            # Iterate over all angle possibilities
            for i, ang in enumerate(self.angs):
                ang = np.array(ang)
                ang_fluxes[i] = self.get_ang_flux(group_id, source, ang, i, phi_prev, eig_bool)
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
            if isinstance(self.op, Diffusion):
                phi = self.get_scalar_flux(group_id, source, phi_prev, eig_bool)
            else:
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
        if isinstance(self.op, Diffusion):
            return phi
        else:
            return phi, ang_fluxes

    def solve_outer(self, source, eig_bool, max_iter=50, tol=1e-2):
        phis = np.ones((self.num_groups, self.num_nodes))
        ang_fluxes = np.zeros((self.num_groups, 4, self.num_nodes))
        for it_count in range(max_iter):
            if self.num_groups != 1:
                print("Gauss-Seidel Iteration: ", it_count)
            phis_prev = np.copy(phis)
            for g in range(self.num_groups):
                if isinstance(self.op, Diffusion):
                    phi = self.solve_in_group(source, g, phis, eig_bool)
                    phis[g] = phi
                else:
                    phi, psi = self.solve_in_group(source, g, phis, eig_bool)
                    phis[g] = phi
                    ang_fluxes[g] = psi
            if self.num_groups == 1:
                break
            else:
                res = np.max(np.abs(phis_prev - phis))
                print("GS Norm: ", res)
            if res < tol:
                break
        if isinstance(self.op, Diffusion):
            return phis
        else:
            return phis, ang_fluxes

    def power_iteration(self, max_iter=50, tol=1e-2):
        # Initialize Guesses
        k = 1
        phi = np.ones((self.num_groups, self.num_nodes))
        fiss_collapsed = self.op.compute_collapsed_fission(phi)
        if isinstance(self.op, Diffusion):
            fission_source = np.zeros((self.num_groups, self.num_nodes))
        else:
            fission_source = np.zeros((self.num_groups, 4, self.num_nodes))
        for it_count in range(max_iter):
            print("Eigenvalue Iteration: ", it_count)
            for g in range(self.num_groups):
                if isinstance(self.op, Diffusion):
                    fission_source[g] = self.op.make_fission_source(g, phi)/k
                else:
                    for i, ang in enumerate(self.angs):
                        fission_source[g, i] = self.op.make_fission_source(g, ang, phi)/k
            if isinstance(self.op, Diffusion):
                new_phi = self.solve_outer(fission_source, True)
            else:
                new_phi, psi = self.solve_outer(fission_source, True)
            new_fiss_collapsed = self.op.compute_collapsed_fission(new_phi)
            new_k = k*new_fiss_collapsed/fiss_collapsed
            err_k = np.abs((new_k - k))/np.abs(new_k)
            phi_group_norms = np.array([np.linalg.norm(new_phi[g] - phi[g], ord=2)
                                       /np.linalg.norm(new_phi[g], ord=2)
                                       for g in range(self.num_groups)])
            err_phi = np.linalg.norm(phi_group_norms, ord=2)
            print("Eigenvalue Norm: ", err_k)
            if err_k < tol and err_phi < tol:
                break
            print("phi, new_phi", np.max(phi), np.max(new_phi))
            print("k, new_k", k, new_k)
            phi = new_phi
            k = new_k
            fiss_collapsed = new_fiss_collapsed
        print("k-effective: ", k)
        if isinstance(self.op, Diffusion):
            return phi, k
        else:
            return phi, psi, k

    def solve(self, source, eigenvalue=False):
        if eigenvalue:
            if isinstance(self.op, Diffusion):
                phis, eigenvalue = self.power_iteration()
                return phis, eigenvalue
            else:
                phis, ang_fluxes, k = self.power_iteration()
                return phis, ang_fluxes, k
        else:
            eig_bool = False
            if isinstance(self.op, Diffusion):
                phis = self.solve_outer(source, eig_bool)
                return phis
            else:
                phis, ang_fluxes = self.solve_outer(source, eig_bool)
                return phis, ang_fluxes
