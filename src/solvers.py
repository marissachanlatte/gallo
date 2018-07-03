import itertools as itr

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
import matplotlib.tri as tri

from formulations.diffusion import Diffusion
from formulations.nda import NDA
from formulations.saaf import SAAF
from upscatter_acceleration import UA

class Solver():
    def __init__(self, operator):
        self.op = operator
        self.ua_bool = False
        if isinstance(self.op, NDA):
            self.ho_op = SAAF(self.op.fegrid, self.op.mat_data)
        self.mat_data = self.op.mat_data
        self.num_groups = self.op.num_groups
        self.num_nodes = self.op.num_nodes
        if not isinstance(self.op, Diffusion) and not isinstance(self.op, NDA):
            self.angs = self.op.angs

    def get_ang_flux(self, group_id, source, ang, angle_id, phi_prev, eig_bool):
        lhs = self.op.make_lhs(ang, group_id)
        rhs = self.op.make_rhs(group_id, source, ang, angle_id, phi_prev, eigenvalue=eig_bool)
        ang_flux = linalg.cg(lhs, rhs)[0]
        return ang_flux

    def get_scalar_flux(self, group_id, source, phi_prev, eig_bool, ho_sols=None):
        scalar_flux = 0
        if isinstance(self.op, Diffusion) or isinstance(self.op, NDA):
            lhs = self.op.make_lhs(group_id, ho_sols=ho_sols)
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
                       tol=1e-3, verbose=True):
        num_mats = self.mat_data.get_num_mats()
        for mat in range(num_mats):
            scatmat = self.mat_data.get_sigs(mat)
            if np.count_nonzero(scatmat) == 0:
                scattering = False
            else:
                scattering = True
        if self.num_groups > 1 and verbose:
            print("Starting Group ", group_id)
        if isinstance(self.op, NDA):
            # Run preliminary solve on low-order system
            ho_sols = 0
            phi_prev[group_id] = self.get_scalar_flux(group_id, source, phi_prev, eig_bool, ho_sols=ho_sols)
        for i in range(max_iter):
            if scattering and verbose:
                print("Within-Group Iteration: ", i)
            if isinstance(self.op, NDA):
                ho_solver = Solver(self.ho_op)
                ho_phis, ho_psis = ho_solver.get_scalar_flux(group_id, source, phi_prev, eig_bool)
                ho_sols = [ho_phis, ho_psis]
                phi = self.get_scalar_flux(group_id, source, phi_prev, eig_bool, ho_sols=ho_sols)
            elif isinstance(self.op, Diffusion):
                phi = self.get_scalar_flux(group_id, source, phi_prev, eig_bool)
            else:
                phi, ang_fluxes = self.get_scalar_flux(group_id, source, phi_prev, eig_bool)
            if not scattering:
                break
            norm = np.linalg.norm(phi - phi_prev[group_id], 2)
            if verbose: print("Norm: ", norm)
            if norm < tol:
                break
            phi_prev[group_id] = np.copy(phi)
        if i == max_iter:
            print("Warning: maximum number of iterations reached in solver")
        if self.num_groups > 1 and verbose:
            print("Finished Group ", group_id)
        if scattering and verbose:
            print("Number of Within-Group Iterations: ", i + 1)
            print("Final Phi Norm: ", norm)
        if isinstance(self.op, Diffusion) or isinstance(self.op, NDA):
            return phi
        else:
            return phi, ang_fluxes

    def solve_outer(self, source, eig_bool, verbose=True, max_iter=50, tol=1e-4):
        phis = np.ones((self.num_groups, self.num_nodes))
        ang_fluxes = np.zeros((self.num_groups, 4, self.num_nodes))
        for it_count in range(max_iter):
            if self.num_groups != 1 and verbose:
                print("Gauss-Seidel Iteration: ", it_count)
            phis_prev = np.copy(phis)
            for g in range(self.num_groups):
                if isinstance(self.op, Diffusion) or isinstance(self.op, NDA):
                    phi = self.solve_in_group(source, g, phis, eig_bool, verbose=verbose)
                    phis[g] = phi
                else:
                    phi, psi = self.solve_in_group(source, g, phis, eig_bool)
                    phis[g] = phi
                    ang_fluxes[g] = psi
            if self.num_groups == 1:
                break
            else:
                if self.ua_bool:
                    # Calculate Correction Term
                    print("Calculating Upscatter Acceleration Term")
                    upscatter_accelerator = UA(self.op, phis, phis_prev, ho_sols)
                    epsilon = upscatter_accelerator.calculate_correction()
                    phis += epsilon
                res = np.max(np.abs(phis_prev - phis))
                if verbose:
                    print("GS Norm: ", res)
            if res < tol:
                break
        if isinstance(self.op, Diffusion) or isinstance(self.op, NDA):
            return phis
        else:
            return phis, ang_fluxes

    def solve(self, source, eigenvalue=False, ua_bool=False):
        if ua_bool:
            self.ua_bool = True
        if eigenvalue:
            if isinstance(self.op, Diffusion) or isinstance(self.op, NDA):
                phis, eigenvalue = self.power_iteration()
                return phis, eigenvalue
            else:
                phis, ang_fluxes, k = self.power_iteration()
                return phis, ang_fluxes, k
        else:
            eig_bool = False
            if isinstance(self.op, Diffusion) or isinstance(self.op, NDA):
                phis = self.solve_outer(source, eig_bool)
                return phis
            else:
                phis, ang_fluxes = self.solve_outer(source, eig_bool)
                return phis, ang_fluxes
