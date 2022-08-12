import numpy as np
import pandas as pd
from rotation_func import *

class sys_info:
    def __init__(self,folder,molecule,variation):
        self.xyz, self.header = import_coord(f'{folder}xtbopt.xyz')
        self.hessian_import = import_hessian(f'{folder}/hessian',self.xyz)
        self.hessian = self.hessian_import.copy()
        self.dipm = import_dipm(f'{folder}/xyz_dipm.csv').iloc[:,:-3]
        self.grad = None#import_gradient(f'{folder}/gradient',self.xyz)

        self.N_atoms = len(self.xyz['atoms'])

        self.angle_list  = [np.linspace(0.0,270.0,4),np.linspace(0.0,270.0,4)]
        
        self.features = Feature(f'{folder}/ml_feature.csv',self.angle_list,grad=self.grad)

        self.init_R_MI,self.xyz = (calc_R(self.xyz))

        #self.H_approx = H_Approx(self.grad)
        #self.H_delta = H_Delta(self.H_approx,self.hessian)

        self.init_P_MI = (rotM_hess(self.init_R_MI,self.xyz))

        self.coord_state = ['init','inert']
        self.molecule = molecule
        self.variation = variation

        self.R_MI_APF_mat = np.zeros([3*self.N_atoms,3*self.N_atoms])
        self.H_APF_mat = np.zeros([3*self.N_atoms,3*self.N_atoms])

        self.H_AA_vec = []
        self.H_AB_vec = []

        self.lambd_len = None
        self.H_pred = np.zeros([3*self.N_atoms,3*self.N_atoms])
        self.H_pred_lambd = None
        self.hessian_lambd = None

    def clear_feature_vec(self):
        self.H_AA_vec = []
        self.H_AB_vec = []
        return

    def clear_all(self):
        self.xyz = None
        self.header = None
        self.hessian_import = None
        self.hessian = None
        self.dipm = None
        self.grad = None#import_gradient(f'{folder}/gradient',self.xyz)

        self.N_atoms =None
        self.angle_list  = None
        
        self.features = None

        self.init_R_MI =None
        #self.H_approx = H_Approx(self.grad)
        #self.H_delta = H_Delta(self.H_approx,self.hessian)

        self.init_P_MI = None

        self.coord_state = None
        self.molecule = None
        self.variation = None

        self.R_MI_APF_mat = None
        self.H_APF_mat = None
        self.H_AA_vec = None
        self.H_AB_vec = None

        self.lambd_len = None
        self.H_pred = None
        self.H_pred_lambd = None
        self.hessian_lambd = None
        return 

    def rot_init_inert(self):
        self.coord_state[0], self.coord_state[1] = self.coord_state[1],self.coord_state[0]

        self.init_R_MI = np.transpose(self.init_R_MI)
        self.init_P_MI = np.transpose(self.init_P_MI)

        self.xyz = coord_rot(self.xyz,self.init_R_MI)


        #self.xyz.sort_values(['x','y','z'],key=abs)

        self.dipm = coord_rot(self.dipm,self.init_R_MI)

        self.hessian = matmul(matmul(self.init_P_MI,self.hessian),np.transpose(self.init_P_MI)) ##Change Hessian 2 times

        #self.H_approx = matmul(matmul(self.init_P_MI,self.H_approx),np.transpose(self.init_P_MI))
        return


    def rot_inert_apf(self):
        if self.coord_state[0] == 'init':
            print('Molecule is not in the right coordinate system')

        elif self.coord_state[0] == 'inert': 
            for atom_A in range(self.N_atoms):
                for atom_B in range(atom_A,self.N_atoms):
                    R_MI_APF = get_R_euler(self.xyz,self.dipm,atom_A,atom_B)
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


    def gen_Hessian_vector(self,train_rot=None,train_set=None):

        if train_rot == True:
            angle_list = self.angle_list
        else:
            angle_list = [[0.0],[0.0]]

        for angle in angle_list[0]:
            rot_Mat = rot_X(angle/360*2*np.pi)
            for atom_A in range(self.N_atoms):
                H_APF =  self.H_APF_mat[3*atom_A:3*atom_A+3,3*atom_A:3*atom_A+3]
                H_APF = matmul(matmul((rot_Mat),H_APF),np.transpose(rot_Mat))
                for i in range(3):
                    for j in range(3):
                        self.H_AA_vec.extend([H_APF[i,j]])

        for angle in angle_list[1]:
            for atom_A in range(self.N_atoms):
                rot_Mat = rot_X(angle/360*2*np.pi)

                for atom_B in range(atom_A+1,self.N_atoms):
                    H_APF =  self.H_APF_mat[3*atom_A:3*atom_A+3,3*atom_B:3*atom_B+3].copy()

                    if [atom_A,atom_B] in self.features.transpose_list:
                        #
                        H_APF = matmul(matmul(rot_X(np.pi),np.transpose(H_APF)),np.transpose(rot_X(np.pi)))

                    H_APF = matmul(matmul((rot_Mat),H_APF),np.transpose(rot_Mat))
                    for i in range(3):
                        for j in range(3):
                            self.H_AB_vec.extend([H_APF[i,j]])

                    if train_set==True:
                        H_APF_T = np.transpose(H_APF)
                        for i in range(3):
                            for j in range(3):
                                self.H_AB_vec.extend([H_APF_T[i,j]])
        return


    def gen_Feature(self,label=None,train_rot=None,train_set=None,data_analysis=None):
        #self.features.get_Feature_homonuclear(R_MI_APF_mat=self.R_MI_APF_mat,N_atoms=self.N_atoms)
        self.features.Feature_AA = []
        self.features.Feature_AB = []
        self.features.get_Feature_heteronuclear(R_MI_APF_mat=self.R_MI_APF_mat,N_atoms=self.N_atoms,xyz = self.xyz,init_R_MI=self.init_R_MI,train_rot=train_rot,train_set=train_set,data_analysis=data_analysis)
        self.features.get_Feature_homonuclear(R_MI_APF_mat=self.R_MI_APF_mat,N_atoms=self.N_atoms,init_R_MI=self.init_R_MI,train_rot=train_rot,data_analysis=data_analysis)

        return self.features.Feature_AA, self.features.Feature_AB

    def get_pred_hessian(self,hessian_homo=None,hessian_hetero=None,check=None):
        if check == True:
            self.H_pred = self.hessian
        else:
            ite_homo = 0
            ite_hetero = 0
            for atom_A in range(self.N_atoms):
                self.H_pred = fill_matrix_block(hessian_homo,self.H_pred,R_mat=self.R_MI_APF_mat,A=atom_A,ite=ite_homo)
                ite_homo += 1

                transpose = None
                for atom_B in range(atom_A+1,self.N_atoms):
                    if [atom_A,atom_B] in self.features.transpose_list:
                        transpose = True
                    self.H_pred = fill_matrix_block(hessian_hetero,self.H_pred,R_mat=self.R_MI_APF_mat,A=atom_A,B=atom_B,ite=ite_hetero,transpose=transpose)
                    ite_hetero +=1
            self.H_pred += self.hessian ## Change Hessian
        return

    def perm_Hess(self,A,B):
        P = np.identity(len(self.H_pred))
        i = 3*A
        j = 3*B

        #P[j:j+3,i:i+3],P[i:i+3,i:i+3]  = P[j:j+3,i:i+3],P[j:j+3,i:i+3]
        P[j:j+3,i:i+3] = P[i:i+3,i:i+3]
        P[i:i+3,i:i+3] = P[i:i+3,j:j+3]
        P[i:i+3,j:j+3] = P[j:j+3,j:j+3]
        P[j:j+3,j:j+3] = P[i:i+3,i:i+3]

        #P[j:j+3,j:j+3],P[j:j+3,i:i+3]  = P[j:j+3,i:i+3],P[j:j+3,j:j+3]
        self.H_pred = matmul(matmul(P,self.H_pred),P)
        return 

    def get_coord_state(self):
        return self.coord_state[0]

    def project_hessian(self,label=None):
        if label=='xTB':
            idx_list,lamb,Q = find_trans_rot(self.hessian,self.xyz)

            self.hessian_lambd = lamb 

            for i in idx_list:
                    i = int(i)
                    self.hessian -= lamb[i] * np.outer(Q.T[i], Q.T[i].T)
                    
        elif label=='pred':
            idx_list,lamb,Q = find_trans_rot(self.H_pred,self.xyz)

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

                    mass_n = elements_dict[atoms[k]]
                    mass_m = elements_dict[atoms[l]]

                    self.hessian[3*k:3*k+3,3*l:3*l+3] =  1/np.sqrt(mass_n*mass_m*mass_unit_in_au**2)*self.hessian[3*k:3*k+3,3*l:3*l+3]

        elif label=='pred':
            for k in range(len(atoms)):
                for l in range(len(atoms)):

                    mass_n = elements_dict[atoms[k]]
                    mass_m = elements_dict[atoms[l]]

                    self.H_pred[3*k:3*k+3,3*l:3*l+3] =  1/np.sqrt(mass_n*mass_m*mass_unit_in_au**2)*self.H_pred[3*k:3*k+3,3*l:3*l+3]
        else:
            print('No hessian was weighted')
        return 

    def gen_eigenvalues(self):
        lambd_temp,Q = linalg.eigh(self.H_pred)
        self.H_pred_lambd =sorted(lambd_temp,key=abs)[self.lambd_len:]

        lambd_temp,Q = linalg.eigh(self.hessian)
        self.hessian_lambd =sorted(lambd_temp,key=abs)[self.lambd_len:]
        return

    def clear(self):
        self.H_AA_vec = []
        self.H_AB_vec = []

        self.lambd_len = None
        self.H_pred = np.zeros([3*self.N_atoms,3*self.N_atoms])

        self.H_pred_lambd = None
        self.hessian_lambd = None
        
        self.hessian = self.hessian_import.copy()
        #self.H_approx = H_Approx(self.grad)
        #self.H_delta = H_Delta(self.H_approx,self.hessian)

        self.features.Feature_AB = []
        self.features.Feature_AA = []

        self.features.transpose_list = []
        self.features.check_list = []

        return


class Feature:
    def __init__(self,folder,angle_list,grad):
        self.folder =folder 
        GFN2_quantities = pd.read_csv(f'{folder}')
        self.CN = np.array(GFN2_quantities.loc[:,['coordination number','delta coordination number']].values.tolist())
        self.dipm_atom = np.array(GFN2_quantities.loc[:,['dipm_atom_x','dipm_atom_y','dipm_atom_z']].values.tolist())
        self.dipm_delta = np.array(GFN2_quantities.loc[:,['dipm_delta_x','dipm_delta_y','dipm_delta_z']].values.tolist())
        self.dipm_only_mull = np.array(GFN2_quantities.loc[:,['delta dipm only mull x','delta dipm only mull y','delta dipm only mull z']].values.tolist())
        self.qm_atom = qm_matrix(np.array(GFN2_quantities.loc[:,['qm_atom_xx','qm_atom_yy', 'qm_atom_zz','qm_atom_xy','qm_atom_xz','qm_atom_yz']].values.tolist()))
        self.qm_delta = qm_matrix(np.array(GFN2_quantities.loc[:,['qm_delta_xx','qm_delta_yy', 'qm_delta_zz','qm_delta_xy','qm_delta_xz','qm_delta_yz']].values.tolist()))
        self.energy_based = np.array(GFN2_quantities.loc[:,['gap (eV)','chem.pot (eV)','HOAO (eV)','LUAO (eV)',
                                    'E_repulsion','E_EHT',' E_disp_2','E_disp_3','E_ies_ixc','E_aes',' E_tot',
                                    'E_axc',' chem_pot_ext','e_gap_ext','ehoao_ext','eluao_ext']].values.tolist())
        #self.grad = grad

        self.names = GFN2_quantities.columns.tolist()
        self.transpose_list = []
        self.check_list = []
        self.Feature_AB = []
        self.Feature_AA = []
        self.angle_list  = angle_list

    def  get_Feature_heteronuclear(self, label=None, R_MI_APF_mat=None, N_atoms = None,xyz = None,init_R_MI=None,train_rot=None,train_set=None,data_analysis=None):

        if self.Feature_AB == []:
            index = [[2,0,0],
                    [1,1,0],
                    [1,0,1],
                    [1,1,0],
                    [0,2,0],
                    [0,1,1],
                    [1,0,1],
                    [0,1,1],
                    [0,0,2]]

            index_old = [[2,0,0],
                        [1,-1,0],
                        [1,0,-1],
                        [-1,1,0],
                        [0,2,0],
                        [0,1,-1],
                        [-1,0,1],
                        [0,-1,1],
                        [0,0,2]]

            if train_rot==True:
                angle_list = self.angle_list

            else:
                angle_list = [[0.0],[0.0]]

            for angle in angle_list[1]:
                rot_Mat = rot_X(angle/360*2*np.pi)

                self.transpose_list = []
                self.check_list = []

                for atom_A in range(N_atoms):
                    for atom_B in range(atom_A+1,N_atoms):

                        A = atom_A
                        B = atom_B

                        i0 = 3*A
                        i3 = 3*A + 3
                        j0 = 3*B 
                        j3 = 3*B + 3

                        self.check_list.append([A,B])

                        if linalg.norm(self.dipm_atom[A]) < linalg.norm(self.dipm_atom[B]):
                            B,A = A,B
                            self.transpose_list.append([B,A])
                            rot_Mat = rot_X(np.pi)#matmul(rot_X(np.pi),rot_Mat)
                        

                        R_MI_APF = R_MI_APF_mat[i0:i3,j0:j3]

                        R_MI_APF = matmul(rot_Mat,R_MI_APF)

                        R_AB = linalg.norm(xyz.iloc[A,1:] - xyz.iloc[B,1:])

                        Quantity_AB = [[],[]]

                        vector_of_ones = np.array([1,1,1])

                        j = 0

                        for atom in [A,B]:
                            
                            dipm_atom = matmul(init_R_MI,self.dipm_atom[atom])
                            dipm_delta = matmul(init_R_MI,self.dipm_delta[atom])
                            dipm_only_mull = matmul(init_R_MI,self.dipm_only_mull[atom])
                                
                            qm_atom = matmul(matmul(init_R_MI,self.qm_atom[atom]),np.transpose(init_R_MI))
                            qm_delta = matmul(matmul(init_R_MI,self.qm_delta[atom]),np.transpose(init_R_MI))

                            dipm_atom = matmul(R_MI_APF,dipm_atom)
                            dipm_delta = matmul(R_MI_APF,dipm_delta)
                            dipm_only_mull = matmul(R_MI_APF,dipm_only_mull)

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

                        Quantity_AB_arr =(np.array(Quantity_AB))

                        Feature_Arith = (Quantity_AB_arr[0] + Quantity_AB_arr[1])/2
                        Feature_Prod = (Quantity_AB_arr[0] * Quantity_AB_arr[1])
                        Feature_AbsDiff = np.abs(Quantity_AB_arr[0] - Quantity_AB_arr[1])

                        idx_len = 9
                        if data_analysis == True:
                            idx_len = 1

                        for idx in range(idx_len):
                            Features = []

                            if idx in [3,6,7]:
                                Features.extend((Quantity_AB[1]))
                                Features.extend((Quantity_AB[0]))
                            else:
                                Features.extend((Quantity_AB[0]))
                                Features.extend((Quantity_AB[1]))
                        
                            Features.extend(Feature_Arith)
                            Features.extend(Feature_Prod)
                            Features.extend(Feature_AbsDiff)

                            Features.extend(index[idx])
                            Features.extend([R_AB])

                            self.Feature_AB.append(Features)
        return





    def get_Feature_homonuclear(self, R_MI_APF_mat=None, N_atoms = None,init_R_MI = None,train_rot=None,data_analysis=None):
        if self.Feature_AA == []:
            index = [[2,0,0],
                    [1,1,0],
                    [1,0,1],
                    [1,1,0],
                    [0,2,0],
                    [0,1,1],
                    [1,0,1],
                    [0,1,1],
                    [0,0,2]]

            if train_rot==True:
                angle_list = self.angle_list

            else:
                angle_list = [[0.0],[0.0]]

            for angle in angle_list[0]:
                rot_Mat = rot_X(angle/360*2*np.pi)

                for A in range(N_atoms):

                    i0 = 3*A
                    i3 = 3*A + 3

                    R_MI_APF = R_MI_APF_mat[i0:i3,i0:i3]
                    R_MI_APF = matmul(rot_Mat,R_MI_APF)

                    Quantity_A = []

                    vector_of_ones = np.array([1,1,1])
                    dipm_atom = matmul(init_R_MI,self.dipm_atom[A])
                    dipm_delta = matmul(init_R_MI,self.dipm_delta[A])
                    dipm_only_mull = matmul(init_R_MI,self.dipm_only_mull[A])

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
                    #Quantity_A.extend(dipm_delta)
                    #Quantity_A.extend(dipm_only_mull)

                    Quantity_A.extend(qm_atom)
                    #Quantity_A.extend(qm_delta)

                    Quantity_A.extend(self.energy_based[A])
                    #Quantity_A.extend(gradient)
                    idx_len = 9
                    if data_analysis == True:
                        idx_len = 1
                    for i in range(idx_len):
                        Features = []
                        Features.extend(np.abs(np.array(Quantity_A)))
                        Features.extend(index[i])

                        self.Feature_AA.append(Features)
        return



def qm_matrix(qm_atom):
    qm_matrix_list = []
    for i in range(len(qm_atom)):
        xx = qm_atom[i,0]
        yy = qm_atom[i,1]
        zz = qm_atom[i,2]
        xy = qm_atom[i,3]
        xz = qm_atom[i,4]
        yz = qm_atom[i,5]

        qm_matrix = np.array([[xx,xy,xz],
                              [xy,yy,yz],
                              [xz,yz,zz]])

        qm_matrix_list.append(qm_matrix)

    return qm_matrix_list


def fill_matrix_block(vector,matrix,R_mat=None,A=None,B=None,ite=None,transpose=None):
    
    if B == None:
        B = A

    k = 9*ite
    for i in range(3):
        for j in range(3):
            matrix[3*A+i,3*B+j] = vector[k]
            #print(f'{k}:{vector[k]}')
            k+=1

    matrix[3*A:3*A+3,3*B:3*B+3] = matmul(matmul(np.transpose(R_mat[3*A:3*A+3,3*B:3*B+3]),matrix[3*A:3*A+3,3*B:3*B+3]),(R_mat[3*A:3*A+3,3*B:3*B+3]))
    
    if transpose == True:
        matrix[3*A:3*A+3,3*B:3*B+3] = matmul(matmul(np.transpose(rot_X(np.pi)),np.transpose(matrix[3*A:3*A+3,3*B:3*B+3])),(rot_X(np.pi)))

    if A != B:
        matrix[3*B:3*B+3,3*A:3*A+3] = np.transpose(matrix[3*A:3*A+3,3*B:3*B+3])

    return matrix


def find_trans_rot(hess,coord):
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


def H_Approx(grad):
    return np.outer(grad,grad)

def H_Delta(H_approx,H_exact):
    return H_exact - H_approx 

def H_Exact(H_approx,H_delta):
    return H_approx + H_delta  