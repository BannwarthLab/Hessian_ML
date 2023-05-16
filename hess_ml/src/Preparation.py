import numpy as np
from operator import matmul
from hess_ml.src.Rotation_func import Rotation_Functions

class Preparation(Rotation_Functions):
    def __init__(self):
        super().__init__
        pass

    def rot_inert_apf(self):

        self.R_MI_APF_mat = np.zeros([self.N_atoms*3,self.N_atoms*3])
        self.H_APF_mat = np.zeros([self.N_atoms*3,self.N_atoms*3])

        for atom_A in range(self.N_atoms):
            i0 = 3*atom_A
            i3 = 3*atom_A + 3

            self.H_APF_mat[i0:i3,i0:i3] = self.hessian[i0:i3,i0:i3].copy() ##Change Hessian

            for atom_B in range(atom_A+1,self.N_atoms):
                xyz_temp = self.xyz.copy()
                R_MI_APF = self.get_R_euler(xyz_temp,self.dipm,atom_A,atom_B)

                H_APF = np.zeros([3,3])
                #Generate the final hessian

                i0 = 3*atom_A
                i3 = 3*atom_A + 3
                j0 = 3*atom_B 
                j3 = 3*atom_B + 3

                H_APF = matmul(matmul(R_MI_APF,self.hessian[i0:i3,j0:j3].copy()),np.transpose(R_MI_APF)) ##Change Hessian

                self.R_MI_APF_mat[i0:i3,j0:j3] = R_MI_APF 
                self.H_APF_mat[i0:i3,j0:j3] = H_APF

        return