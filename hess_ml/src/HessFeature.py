import numpy as np
import pandas as pd
from scipy import linalg
from src.Rotation_func import Rotation_Functions
from operator import matmul

class HessFeature(Rotation_Functions):
    def __init__(self):
        Rotation_Functions.__init__


    def  get_Feature_heteronuclear(self):

        index =     [[2,0,0],[1,-1,0],[1,0,-1],
                    [-1,1,0],[0,2,0],[0,1,-1],
                    [-1,0,1],[0,-1,1],[0,0,2]]


        self.transpose_list = []
        self.check_list = []
        self.Feature_AB = []

        for atom_A in range(self.N_atoms):
            for atom_B in range(atom_A+1,self.N_atoms):

                A = atom_A
                B = atom_B

                i0 = 3*A
                i3 = 3*A + 3
                j0 = 3*B 
                j3 = 3*B + 3

                R_MI_APF = self.R_MI_APF_mat[i0:i3,j0:j3]

                self.check_list.append([A,B])

                if linalg.norm(self.dipm_atom[A]) < linalg.norm(self.dipm_atom[B]):
                    B,A = A,B
                    self.transpose_list.append([B,A])

                    rot_Mat = self.rot_X(np.pi)
    
                    R_MI_APF = matmul(rot_Mat,R_MI_APF)

                R_AB = linalg.norm(self.xyz.iloc[A,1:] - self.xyz.iloc[B,1:])

                Quantity_AB = [[],[]]

                vector_of_ones = np.array([1,1,1])

                j = 0

                for atom in [A,B]:
                    
                    dipm_atom = matmul(self.init_R_MI,self.dipm_atom[atom])
                    dipm_delta = matmul(self.init_R_MI,self.dipm_delta[atom])
                    dipm_only_mull = matmul(self.init_R_MI,self.dipm_only_mull[atom])
                    
                    dipm_atom = matmul(R_MI_APF,dipm_atom)
                    dipm_delta = matmul(R_MI_APF,dipm_delta)
                    dipm_only_mull = matmul(R_MI_APF,dipm_only_mull)
                        

                    qm_atom = matmul(matmul(self.init_R_MI,self.qm_atom[atom]),np.transpose(self.init_R_MI))
                    qm_delta = matmul(matmul(self.init_R_MI,self.qm_delta[atom]),np.transpose(self.init_R_MI))

                    qm_atom = matmul(matmul(R_MI_APF,qm_atom),np.transpose(R_MI_APF))
                    qm_delta = matmul(matmul(R_MI_APF,qm_delta),np.transpose(R_MI_APF))

                    qm_atom = matmul(qm_atom,vector_of_ones)
                    qm_delta = matmul(qm_delta,vector_of_ones)

                    #gradient = matmul(R_MI_APF,self.grad[3*atom:3*atom+3])

                    #for k in range(3):
                    #    Quantity_AB[j].extend([np.sum(qm_atom[k,:])])

                    #for k in range(3):
                    #    Quantity_AB[j].extend([np.sum(qm_delta[k,:])])

                    Quantity_AB[j].extend(self.CN[atom])
                    Quantity_AB[j].extend(dipm_atom)
                    Quantity_AB[j].extend(dipm_delta)
                    Quantity_AB[j].extend(dipm_only_mull)

                    Quantity_AB[j].extend(qm_atom)
                    Quantity_AB[j].extend(qm_delta)
                    Quantity_AB[j].extend(self.energy_based[atom])

                    #Quantity_AB[j].extend(gradient)


                    j+=1

                Quantity_AB_arr =np.array(Quantity_AB)

                Feature_Arith = list((Quantity_AB_arr[0] + Quantity_AB_arr[1])/2)
                Feature_Prod = list(Quantity_AB_arr[0] * Quantity_AB_arr[1])
                Feature_AbsDiff = list(np.abs(Quantity_AB_arr[0] - Quantity_AB_arr[1]))

                Features_temp = []
                Features_temp.extend(Quantity_AB[0])
                Features_temp.extend(Quantity_AB[1])
                Features_temp.extend(Feature_Arith)
                Features_temp.extend(Feature_Prod)
                Features_temp.extend(Feature_AbsDiff)

                for idx in range(9):
                    Features_temp2 = Features_temp.copy()
                    Features_temp2.extend(index[idx])
                    Features_temp2.extend([R_AB])

                    self.Feature_AB.append(Features_temp2)


        del Quantity_AB_arr
        del Quantity_AB
        del Features_temp 
        del R_MI_APF

        del Feature_Arith
        del Feature_Prod
        del Feature_AbsDiff
        
        return





    def get_Feature_homonuclear(self):

        index = [[2,0,0],[1,1,0],[1,0,1],
                [1,1,0],[0,2,0],[0,1,1],
                [1,0,1],[0,1,1],[0,0,2]]
                
        self.Feature_AA = []

        for A in range(self.N_atoms):
            rot_Mat = self.rot_X(0.0/360*2*np.pi)

            i0 = 3*A
            i3 = 3*A + 3

            R_MI_APF = self.R_MI_APF_mat[i0:i3,i0:i3]
            R_MI_APF = matmul(rot_Mat,R_MI_APF)

            Quantity_A = []


            vector_of_ones = np.array([1,1,1])

            dipm_atom = matmul(self.init_R_MI,self.dipm_atom[A])
            dipm_delta = matmul(self.init_R_MI,self.dipm_delta[A])
            dipm_only_mull = matmul(self.init_R_MI,self.dipm_only_mull[A])

            dipm_atom = matmul(R_MI_APF,self.dipm_atom[A])
            dipm_delta = matmul(R_MI_APF,self.dipm_delta[A])
            dipm_only_mull = matmul(R_MI_APF,self.dipm_only_mull[A])

            qm_atom = matmul(matmul(R_MI_APF,self.qm_atom[A]),np.transpose(R_MI_APF))
            qm_delta = matmul(matmul(R_MI_APF,self.qm_delta[A]),np.transpose(R_MI_APF))
            #gradient = matmul(R_MI_APF,self.grad[3*A:3*A+3])

            #for i in range(3):
            #   Quantity_A.extend([np.sum(qm_atom[i,:])])
            qm_atom = matmul(qm_atom,vector_of_ones)
            qm_delta = matmul(qm_delta,vector_of_ones)

            #for i in range(3):
            #    Quantity_A.extend([np.sum(qm_delta[i,:])])

            Quantity_A.extend(self.CN[A])

            Quantity_A.extend(dipm_atom)
            Quantity_A.extend(dipm_delta)
            Quantity_A.extend(dipm_only_mull)

            Quantity_A.extend(qm_atom)
            Quantity_A.extend(qm_delta)

            Quantity_A.extend(self.energy_based[A])

            #Quantity_A.extend(gradient)
            
            if self.diag == 'DTR':
                for i in range(9):
                    Features_temp = []
                    Features_temp.extend(np.abs(np.array(Quantity_A)))
                    Features_temp.extend(index[i])

                    self.Feature_AA.append(Features_temp)

            if self.diag == 'GNN':
                Features_temp = []
                Features_temp.extend(np.abs(np.array(Quantity_A)))
                self.Feature_AA.append(Features_temp)


        del Quantity_A
        
        return






