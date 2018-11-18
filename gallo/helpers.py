import numpy as np

class Helper():
    def __init__(self, grid, mat_data):
        self.fegrid = grid
        self.mat_data = mat_data
        self.num_groups = self.mat_data.get_num_groups()
        self.num_elts = self.fegrid.num_elts

    def flux_at_elt(self, flux):
        """ Takes in fluxes at nodes and returns averaged fluxes for each element. """
        num_groups = np.shape(flux)[0]
        elt_flux = np.zeros((num_groups, self.num_elts))
        for g in range(num_groups):
            for ele in range(self.num_elts):
                vertices = self.fegrid.element(ele).vertices
                func = 0
                for vtx in vertices:
                    func += flux[g, vtx]
                func /= 3
                elt_flux[g, ele] = func
        return elt_flux

    def integrate_flux(self, flux):
        # Integrate Flux Over Total Domain
        areas = np.array([self.fegrid.element_area(ele) for ele in range(self.num_elts)])
        elt_fluxes = self.flux_at_elt(flux)
        return np.sum(areas*elt_fluxes)

    def make_full_fission_source(self, phi):
        fiss_source = np.zeros((self.num_groups, self.num_elts))
        flux_at_elt = self.flux_at_elt(phi)
        for group in range(self.num_groups):
            for elt in range(self.num_elts):
                midx = self.fegrid.element(elt).mat_id
                phi = flux_at_elt[:, elt]
                fiss_source[group, elt] = self.compute_fission_source(midx, phi, group)
        return fiss_source

    def compute_scattering_source(self, midx, phi, group_id):
        scatmat = self.mat_data.get_sigs(midx)
        ssource = 0
        for g_prime in range(self.num_groups):
            if g_prime != group_id:
                ssource += scatmat[g_prime, group_id]*phi[g_prime]
        return ssource

    def compute_fission_source(self, midx, phi, group_id):
        fsource = 0
        chi = self.mat_data.get_chi(midx, group_id)
        for g_prime in range(self.num_groups):
            nu = self.mat_data.get_nu(midx, g_prime)
            sigf = self.mat_data.get_sigf(midx, g_prime)
            fsource += nu*sigf*phi[g_prime]
        fsource *= chi
        return fsource
