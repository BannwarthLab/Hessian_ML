import numpy as np
import pandas as pd
from operator import matmul


class HessTarget:
    
    def __init__(self) -> None:
        pass 

    def gen_Hessian_vector(self,transpose_list):
        self.Target_AA = []
        self.Target_AB = []

        for atom_A in range(self.N_atoms):

            H_APF =  self.H_APF_mat[3*atom_A:3*atom_A+3,3*atom_A:3*atom_A+3]
            self.Target_AA.append(list(H_APF[np.triu_indices(3)]))
        
            for atom_B in range(atom_A+1,self.N_atoms):
                H_APF =  self.H_APF_mat[3*atom_A:3*atom_A+3,3*atom_B:3*atom_B+3].copy()

                if [atom_A,atom_B] in transpose_list:
                    H_APF = matmul(matmul(self.rot_X(np.pi),np.transpose(H_APF)),np.transpose(self.rot_X(np.pi)))

                self.Target_AB.append(list(H_APF.flatten()))
        
        return
    