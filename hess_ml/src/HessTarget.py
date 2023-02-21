import numpy as np
import pandas as pd
from operator import matmul
from scipy import linalg
import os 

class HessTarget:
    
    def __init__(self) -> None:
        pass 

    def gen_Hessian_vector(self,transpose_list):
        self.Target_AA = []
        self.Target_AB = []

        for atom_A in range(self.N_atoms):

            H_APF =  self.H_APF_mat[3*atom_A:3*atom_A+3,3*atom_A:3*atom_A+3]
            self.Target_AA.extend(list(H_APF.flatten()))

            for atom_B in range(atom_A+1,self.N_atoms):
                H_APF =  self.H_APF_mat[3*atom_A:3*atom_A+3,3*atom_B:3*atom_B+3].copy()

                if [atom_A,atom_B] in transpose_list:
                    H_APF = matmul(matmul(self.rot_X(np.pi),np.transpose(H_APF)),np.transpose(self.rot_X(np.pi)))

                self.Target_AB.extend(list(H_APF.flatten()))
        
        return

    def gen_Hessian_GNN(self):
        self.Target_AA = np.zeros([self.N_atoms,6])
        for atom_A in range(self.N_atoms):
            self.Target_AA[atom_A] =  self.Hess_to_vec(self.H_APF_mat[3*atom_A:3*atom_A+3,3*atom_A:3*atom_A+3])

        return

    def Hess_to_vec(self,hess):

        vec = np.zeros(6)
        vec[:3] = hess[0,:3]
        vec[3:5] = hess[1,1:]
        vec[5] = hess[2,2]

        return vec
#Maybe at other point --> at predict or something

'''    def get_pred_hessian(self,hessian_homo=None,hessian_hetero=None,check=None):
        if check == True:
            self.H_pred = self.hessian

        ite_homo = 0
        ite_hetero = 0
        for atom_A in range(self.N_atoms):
            self.H_pred = self.fill_matrix_block(hessian_homo,self.H_pred,R_mat=self.R_MI_APF_mat,A=atom_A,ite=ite_homo)
            ite_homo += 1

            transpose = None
            for atom_B in range(atom_A+1,self.N_atoms):

                if [atom_A,atom_B] in self.transpose_list:
                    transpose = True
                self.H_pred = self.fill_matrix_block(hessian_hetero,self.H_pred,R_mat=self.R_MI_APF_mat,A=atom_A,B=atom_B,ite=ite_hetero,transpose=transpose)
                ite_hetero +=1
            #self.H_pred += self.H_approx ## Change Hessian
        return'''
