import numpy as np
from scipy import linalg
from hess_ml.src.constants import Const
from operator import matmul
from hess_ml.src.Rotation_func import Rotation_Functions

class Observables(Rotation_Functions,Const):
    def __init__(self) -> None:
        super().__init__
        pass

    def get_coord_state(self):
        return self.coord_state[0]

    def gen_Frequencies(self,hess_vec_aa,hess_vec_ab,pred=False): 
        
        if pred == True:
            Hessian = self.gen_hess_from_vec_pred(hess_vec_aa,hess_vec_ab)
        else:
            Hessian = self.gen_hess_from_vec_true(hess_vec_aa,hess_vec_ab)
        
        Hess_prj,lamb_len = self.project_hessian(Hessian.copy())
        
        Hess_prj_wgt = self.weight_hessian(Hess_prj)

        lambd_temp,Q = linalg.eigh(Hess_prj_wgt)
        
        eigv = lambd_temp

        freq = np.zeros(len(eigv))
        
        for i in range(len(eigv)):

            if eigv[i] >= 0.0:
                freq[i] = np.sqrt(eigv[i])

            else:
                freq[i] = -np.sqrt(abs(eigv[i]))

        return freq,Hessian

    def get_partition_func(self,freq):

        Z = 1
        
        for i in range(len(freq)):
            
            if freq[i] > 0.0:
                Z *= (1 - np.exp(-freq[i]*Const.conv_Eh_to_J/(Const.boltzmann_const*298.15)))**-1

        return Z
    
    def get_ZPE(self,freq): 

        ZPE = 1/2*np.sum(freq)

        return ZPE
    
    def get_harmonic_ZPE(self,freq): # in kJ/mol
        freq = freq.copy()
        for i in range(len(freq)):
            if freq[-i] == 0.0:
                freq.pop(-i)

        ZPE = 1/2*np.sum(1/np.array(freq))**-1*0.01196265919

        return ZPE  
    
    def project_hessian(self,Hessian):
        idx_list,lamb,Q = self.find_trans_rot(Hessian,self.xyz)
        lamb_len = len(idx_list)
        for i in idx_list:
                i = int(i)
                Hessian -= lamb[i] * np.outer(Q.T[i], Q.T[i].T)

        return Hessian,lamb_len
    

    def weight_hessian(self,Hessian):
        atoms = self.xyz['atoms']

        for k in range(len(atoms)):
            for l in range(k,len(atoms)):
                k3 = 3*k 
                l3 = 3*l
                
                mass_n = Const.elements_dict[atoms[k]]
                mass_m = Const.elements_dict[atoms[l]]

                Hessian[k3:k3+3,l3:l3+3] =  1/np.sqrt(mass_n*mass_m*Const.mass_unit_in_au**2)*Hessian[k3:k3+3,l3:l3+3]
                if k != l:
                    Hessian[l3:l3+3,k3:k3+3] = Hessian[k3:k3+3,l3:l3+3]

        return Hessian


    def fill_matrix_block_AB(self,vector,matrix,R_mat=None,A=None,B=None,ite=None,transpose=False):

        A3 = 3*A 
        B3 = 3*B

        matrix[A3:A3+3,B3:B3+3] = vector.reshape(3,3)

        if transpose == True:
            matrix[A3:A3+3,B3:B3+3] = matmul(matmul(np.transpose(self.rot_X(np.pi)),np.transpose(matrix[A3:A3+3,B3:B3+3])),(self.rot_X(np.pi)))

        matrix[A3:A3+3,B3:B3+3] = matmul(matmul(np.transpose(R_mat[A3:A3+3,B3:B3+3]),matrix[A3:A3+3,B3:B3+3]),(R_mat[A3:A3+3,B3:B3+3]))

        matrix[B3:B3+3,A3:A3+3] = np.transpose(matrix[A3:A3+3,B3:B3+3])

        return matrix
    
    def fill_matrix_block_AA(self,vector,matrix,R_mat=None,A=None,ite=None):

        A3 = 3*A

        temp_mat = np.zeros([3,3])
        temp_mat[np.triu_indices(temp_mat.shape[0],k=0)] = vector
        temp_mat = temp_mat + temp_mat.T - np.diag(np.diag(temp_mat))

        matrix[A3:A3+3,A3:A3+3] = temp_mat

        matrix[A3:A3+3,A3:A3+3] = matmul(matmul(np.transpose(R_mat[A3:A3+3,A3:A3+3]),matrix[A3:A3+3,A3:A3+3]),(R_mat[A3:A3+3,A3:A3+3]))

        return matrix
    
    def find_trans_rot(self,hess,coord):
        Nat = len(coord)

        overlap_mat = np.zeros([6,3*Nat])
        
        trans_x = np.array([1.,0.,0.])
        trans_y = np.array([0.,1.,0.])
        trans_z = np.array([0.,0.,1.])

        for i in range(Nat):
            i3 = 3*i
            overlap_mat[0,i3:i3+3] = trans_x
            overlap_mat[1,i3:i3+3] = trans_y
            overlap_mat[2,i3:i3+3] = trans_z

            overlap_mat[3,i3:i3+3] = np.array([0.,coord.loc[i,'z'],-coord.loc[i,'y']])
            overlap_mat[4,i3:i3+3] = np.array([-coord.loc[i,'z'],0.,coord.loc[i,'x']])
            overlap_mat[5,i3:i3+3] = np.array([coord.loc[i,'y'],-coord.loc[i,'x'],0.])

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


    def gen_hess_from_vec_pred(self,hess_vec_aa,hess_vec_ab):
        ite_homo = 0
        ite_hetero = 0
        Hessian = np.zeros([self.N_atoms*3,self.N_atoms*3])

        for atom_A in range(self.N_atoms):
            #Hessian = self.fill_matrix_block_AA(hess_vec_aa[atom_A],Hessian,R_mat=self.R_MI_APF_mat,A=atom_A)
            #ite_homo += 1

            for atom_B in range(atom_A+1,self.N_atoms):
                transpose = False
                if [atom_A,atom_B] in self.transpose_list:
                    transpose = True
                Hessian = self.fill_matrix_block_AB(hess_vec_ab[ite_hetero],Hessian,R_mat=self.R_MI_APF_mat,A=atom_A,B=atom_B,ite=ite_hetero,transpose=transpose)
                ite_hetero +=1
            #self.H_pred += self.H_approx ## Change Hessian


        for atom_A in range(self.N_atoms):
            for atom_B in range(self.N_atoms):
                if atom_A != atom_B:
                    Hessian[3*atom_A:3*atom_A+3,3*atom_A:3*atom_A+3] -= Hessian[3*atom_A:3*atom_A+3,3*atom_B:3*atom_B+3]

        return Hessian
    

    def gen_hess_from_vec_true(self,hess_vec_aa,hess_vec_ab):
        ite_homo = 0
        ite_hetero = 0
        Hessian = np.zeros([self.N_atoms*3,self.N_atoms*3])

        for atom_A in range(self.N_atoms):
            Hessian = self.fill_matrix_block_AA(hess_vec_aa[atom_A],Hessian,R_mat=self.R_MI_APF_mat,A=atom_A)
            ite_homo += 1

            for atom_B in range(atom_A+1,self.N_atoms):
                transpose = False
                if [atom_A,atom_B] in self.transpose_list:
                    transpose = True
                Hessian = self.fill_matrix_block_AB(hess_vec_ab[ite_hetero],Hessian,R_mat=self.R_MI_APF_mat,A=atom_A,B=atom_B,ite=ite_hetero,transpose=transpose)
                ite_hetero +=1
            #self.H_pred += self.H_approx ## Change Hessian
        return Hessian