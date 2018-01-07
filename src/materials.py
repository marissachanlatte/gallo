import numpy as np
import os
class Materials():
    def __init__(self, filename):
        """Constructor of materials object. Stores material data for all 
        materials """

        # Verify file exists
        assert os.path.exists(filename), "Material file: " + filename\
            + " does not exist"
        with open(filename) as fp:
            line = fp.readline()
            self.num_mats = int(line)
            self.names = []
            self.sig_t = np.zeros(self.num_mats)
            self.sig_a = np.zeros(self.num_mats)
            self.sig_s = np.zeros(self.num_mats)
            self.sig_f = np.zeros(self.num_mats)
            self.D = np.zeros(self.num_mats)
            self.nu= np.zeros(self.num_mats)
            for i in range(self.num_mats):
                line = fp.readline()
                attributes = line.split("|")
                self.names.append(attributes[1].strip())
                self.sig_a[i] = float(attributes[2])
                self.sig_s[i] = float(attributes[3])
                self.sig_f[i] = float(attributes[4])
                self.nu[i] = float(attributes[5])

                # Derived quantities
                self.sig_t[i] = self.sig_a[i] + self.sig_s[i]
                self.D[i] = 1/(3*self.sig_t[i])
                

    def get_name(self, mat_id):
        return self.names[mat_id]

    def get_sigt(self, mat_id):
        return self.sig_t[mat_id]

    def get_siga(self, mat_id):
        return self.sig_a[mat_id]

    def get_sigs(self, mat_id):
        return self.sig_s[mat_id]

    def get_sigf(self, mat_id):
        return self.sig_f[mat_id]

    def get_diff(self, mat_id):
        return self.D[mat_id]

    def get_nu(self, mat_id):
        return self.nu[mat_id]

