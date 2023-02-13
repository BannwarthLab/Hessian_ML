import numpy as np
from scipy import linalg
from constants import Const
from operator import matmul
from Rotation_func import Rotation_Functions
class Observables(Rotation_Functions,Const):
    def __init__(self) -> None:
        super().__init__
        pass

    def get_coord_state(self):
        return self.coord_state[0]

    def project_hessian(self,label=None):
        if label=='xTB':
            idx_list,lamb,Q = self.find_trans_rot(self.hessian,self.xyz)

            self.hessian_lambd = lamb 

            for i in idx_list:
                    i = int(i)
                    self.hessian -= lamb[i] * np.outer(Q.T[i], Q.T[i].T)
                    
        elif label=='pred':
            idx_list,lamb,Q = self.find_trans_rot(self.H_pred,self.xyz)

            self.lambd_len = len(idx_list)

            self.H_pred_lambd = lamb

            for i in idx_list:
                    i = int(i)
                    self.H_pred -= lamb[i] * np.outer(Q.T[i], Q.T[i].T)
        else: 
            print('No hessian was projected')
        return 

    def weight_hessian(self,label=None):
        atoms = self.xyz['atoms']

        if label=='xTB':
            for k in range(len(atoms)):
                for l in range(len(atoms)):

                    mass_n = Const.elements_dict[atoms[k]]
                    mass_m = Const.elements_dict[atoms[l]]

                    self.hessian[3*k:3*k+3,3*l:3*l+3] =  1/np.sqrt(mass_n*mass_m*Const.mass_unit_in_au**2)*self.hessian[3*k:3*k+3,3*l:3*l+3]

        elif label=='pred':
            for k in range(len(atoms)):
                for l in range(len(atoms)):

                    mass_n = Const.elements_dict[atoms[k]]
                    mass_m = Const.elements_dict[atoms[l]]

                    self.H_pred[3*k:3*k+3,3*l:3*l+3] =  1/np.sqrt(mass_n*mass_m*Const.mass_unit_in_au**2)*self.H_pred[3*k:3*k+3,3*l:3*l+3]
        else:
            print('No hessian was weighted')
        return 

    def gen_eigenvalues(self):
        lambd_temp,Q = linalg.eigh(self.H_pred)
        self.H_pred_lambd =sorted(lambd_temp,key=abs)[self.lambd_len:]

        lambd_temp,Q = linalg.eigh(self.hessian)
        self.hessian_lambd =sorted(lambd_temp,key=abs)[self.lambd_len:]
        return


    def fill_matrix_block(self,vector,matrix,R_mat=None,A=None,B=None,ite=None,transpose=None):
        
        if B == None:
            B = A

        k = 9*ite
        for i in range(3):
            for j in range(3):
                matrix[3*A+i,3*B+j] = vector[k]
                #print(f'{k}:{vector[k]}')
                k+=1

        if transpose == True:
            matrix[3*A:3*A+3,3*B:3*B+3] = matmul(matmul(np.transpose(self.rot_X(np.pi)),np.transpose(matrix[3*A:3*A+3,3*B:3*B+3])),(rf.rot_X(np.pi)))

        matrix[3*A:3*A+3,3*B:3*B+3] = matmul(matmul(np.transpose(R_mat[3*A:3*A+3,3*B:3*B+3]),matrix[3*A:3*A+3,3*B:3*B+3]),(R_mat[3*A:3*A+3,3*B:3*B+3]))

        if A != B:
            matrix[3*B:3*B+3,3*A:3*A+3] = np.transpose(matrix[3*A:3*A+3,3*B:3*B+3])

        return matrix


    def find_trans_rot(self,hess,coord):
        Nat = len(coord)

        overlap_mat = np.zeros([6,3*Nat])
        
        trans_x = np.array([1.,0.,0.])
        trans_y = np.array([0.,1.,0.])
        trans_z = np.array([0.,0.,1.])

        for i in range(Nat):
            overlap_mat[0,3*i:3*i+3] = trans_x
            overlap_mat[1,3*i:3*i+3] = trans_y
            overlap_mat[2,3*i:3*i+3] = trans_z

            overlap_mat[3,3*i:3*i+3] = np.array([0.,coord.loc[i,'z'],-coord.loc[i,'y']])
            overlap_mat[4,3*i:3*i+3] = np.array([-coord.loc[i,'z'],0.,coord.loc[i,'x']])
            overlap_mat[5,3*i:3*i+3] = np.array([coord.loc[i,'y'],-coord.loc[i,'x'],0.])

        overlap_mat = overlap_mat / Nat
        overlap_mat = 1/(linalg.norm(overlap_mat,1)) * overlap_mat


        lamb, Q = linalg.eigh(hess)

        M = matmul(overlap_mat,Q)

        norm_x = np.array(linalg.norm(coord.loc[:,'x']))
        
        norm_y = np.array(linalg.norm(coord.loc[:,'y']))

        norm_z = np.array(linalg.norm(coord.loc[:,'z']))

        idx_len = 6 

        if (norm_x + norm_y) < 1e-6 or (norm_y + norm_z) < 1e-6 or (norm_z + norm_x)< 1e-6:
            idx_len = 5

        M_sum = np.zeros(3*Nat)
        for i in range(len(M_sum)):
            M_sum[i] = np.sum(np.abs(M[:,i]))

        idx_list = np.zeros(idx_len)
        for i in range(idx_len):
            idx = np.where(M_sum == np.amax(M_sum))[0][0]
            idx_list[i] = int(idx)
            M_sum[idx] -= M_sum[idx]

        return idx_list,lamb,Q