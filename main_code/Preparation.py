import numpy as np
from operator import matmul

class Preparation:
    def __init__(self):
        pass

    def rot_init_inert(self):

        self.init_R_MI = np.transpose(self.init_R_MI)
        self.init_P_MI = np.transpose(self.init_P_MI)

        self.xyz = self.coord_rot(self.xyz,self.init_R_MI)

        self.dipm = self.coord_rot(self.dipm,self.init_R_MI)

        self.hessian = matmul(matmul(self.init_P_MI,self.hessian),np.transpose(self.init_P_MI)) ##Change Hessian 2 times

        return


    def rot_inert_apf(self):

        self.R_MI_APF_mat = np.zeros([self.N_atoms*3,self.N_atoms*3])
        self.H_APF_mat = np.zeros([self.N_atoms*3,self.N_atoms*3])

        for atom_A in range(self.N_atoms):
            for atom_B in range(atom_A,self.N_atoms):
                R_MI_APF = self.get_R_euler(self.xyz,self.dipm,atom_A,atom_B)
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