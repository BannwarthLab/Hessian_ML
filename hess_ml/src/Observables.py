import numpy as np
from scipy import linalg
from src.constants import Const
from operator import matmul
from src.Rotation_func import Rotation_Functions

class Observables(Rotation_Functions,Const):
    def __init__(self) -> None:
        super().__init__
        pass

    def get_coord_state(self):
        return self.coord_state[0]

    def gen_Frequencies(self,hess_vec_aa,hess_vec_ab): #in cm^-1

        Hessian = self.gen_hess_from_vec(hess_vec_aa,hess_vec_ab)

        #Hess_prj,lamb_len = self.project_hessian(Hessian)
        Hess_prj_wgt = self.weight_hessian(Hessian)
        lambd_temp,Q = linalg.eigh(Hess_prj_wgt)
        eigv = lambd_temp#sorted(lambd_temp,key=abs)[lamb_len:]

        freq = np.zeros(len(eigv))
        
        for i in range(len(eigv)):
            if eigv[i] < 0.0:
                freq[i] = 0 # -np.sqrt(np.abs(eigv[i]))
            else:
                freq[i] =  np.sqrt(eigv[i])

        return freq*219474.63#sorted(freq*219474.63,key=abs)[self.lamb_len:]
    
    def frequency(lamb):
     
     freq_val = np.zeros(len(lamb))

     for i in range(len(lamb)):
          
          if lamb[i] >= 0:
               freq_val[i] = np.sqrt(lamb[i])
          else:
               freq_val[i] = 0

     return freq_val


    def get_ZPE(self,freq): # in kJ/mol

        ZPE = 1/2*np.sum(freq)*0.01196265919

        return ZPE  
    
    def get_harmonic_ZPE(self,freq): # in kJ/mol
        for i in range(len(freq)):
            if freq[-i] == 0.0:
                freq.pop(-i)

        ZPE = 1/2*np.sum(1/np.array(freq))**-1*0.01196265919

        return ZPE  
    
    def project_hessian(self):
                    
        idx_list,lamb,Q = self.find_trans_rot(self.hessian,self.xyz)
        self.lamb_len = len(idx_list)
        for i in idx_list:
                i = int(i)
                self.hessian -= lamb[i] * np.outer(Q.T[i], Q.T[i].T)

        return 

    def weight_hessian(self,Hessian):
        atoms = self.xyz['atoms']

        for k in range(len(atoms)):
            for l in range(len(atoms)):

                mass_n = Const.elements_dict[atoms[k]]
                mass_m = Const.elements_dict[atoms[l]]

                Hessian[3*k:3*k+3,3*l:3*l+3] =  1/np.sqrt(mass_n*mass_m*Const.mass_unit_in_au**2)*Hessian[3*k:3*k+3,3*l:3*l+3]
        return Hessian


    def fill_matrix_block(self,vector,matrix,R_mat=None,A=None,B=None,ite=None,transpose=False):
        
        if B == None:
            B = A

        k = 9*ite
        for i in range(3):
            for j in range(3):
                matrix[3*A+i,3*B+j] = vector[k] #reshape may also be possible
                #print(f'{k}:{vector[k]}')
                k+=1

        if transpose == True:
            matrix[3*A:3*A+3,3*B:3*B+3] = matmul(matmul(np.transpose(self.rot_X(np.pi)),np.transpose(matrix[3*A:3*A+3,3*B:3*B+3])),(self.rot_X(np.pi)))

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


    def gen_hess_from_vec(self,hess_vec_aa,hess_vec_ab):
        ite_homo = 0
        ite_hetero = 0
        Hessian = np.zeros([self.N_atoms*3,self.N_atoms*3])

        for atom_A in range(self.N_atoms):
            Hessian = self.fill_matrix_block(hess_vec_aa,Hessian,R_mat=self.R_MI_APF_mat,A=atom_A,ite=ite_homo)
            ite_homo += 1

            transpose = False
            for atom_B in range(atom_A+1,self.N_atoms):

                if [atom_A,atom_B] in self.transpose_list:
                    transpose = True
                Hessian = self.fill_matrix_block(hess_vec_ab,Hessian,R_mat=self.R_MI_APF_mat,A=atom_A,B=atom_B,ite=ite_hetero,transpose=transpose)
                ite_hetero +=1
            #self.H_pred += self.H_approx ## Change Hessian
        return Hessian