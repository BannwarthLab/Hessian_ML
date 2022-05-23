from mimetypes import read_mime_types
from numpy import indices
import pandas as pd
from rotation_func import *

class sys_info:
    def __init__(self,folder,molecule,variation):
        self.xyz, self.header = import_coord(f'{folder}coord.xyz')
        self.hessian = import_hessian(f'{folder}/hessian',self.xyz)
        self.dipm = import_dipm(f'{folder}/xyz_dipm.csv').iloc[:,:-3]
        self.N_atoms = len(self.xyz['atoms'])
        self.features = Feature(f'{folder}/ml_feature.csv')
        self.init_R_MI = np.transpose(calc_R(self.xyz))
        self.init_P_MI = np.transpose(rotM_hess(self.init_R_MI,self.xyz))
        self.coord_state = ['init','inert']
        self.molecule = molecule
        self.variation = variation
        self.pred_hess = None
        self.R_MI_APF_mat = np.zeros([3*self.N_atoms,3*self.N_atoms])
        self.H_APF_mat = np.zeros([3*self.N_atoms,3*self.N_atoms])

        self.H_AA_vec = []
        self.H_AB_vec = []


    def rot_init_inert(self):
        self.coord_state[0], self.coord_state[1] = self.coord_state[1],self.coord_state[0]
        self.init_R_MI = np.transpose(self.init_R_MI)
        self.init_P_MI = np.transpose(self.init_P_MI)
        self.xyz = coord_rot(self.xyz,self.init_R_MI)
        self.dipm = coord_rot(self.dipm,self.init_R_MI)
        self.hessian = matmul(matmul(self.init_P_MI,self.hessian),np.transpose(self.init_P_MI))
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

                    H_APF = matmul(matmul(R_MI_APF,self.hessian[i0:i3,j0:j3].copy()),(np.transpose(R_MI_APF)))
                    self.R_MI_APF_mat[i0:i3,j0:j3] = R_MI_APF       
                    self.H_APF_mat[i0:i3,j0:j3] = H_APF
        return


    def gen_Hessian_vector(self):

        for atom_A in range(self.N_atoms):
            H_APF =  self.H_APF_mat[3*atom_A:3*atom_A+3,3*atom_A:3*atom_A+3]
            for i in range(3):
                for j in range(3):
                    self.H_AA_vec.extend([H_APF[i,j]])

            for atom_B in range(atom_A+1,self.N_atoms):
                H_APF =  self.H_APF_mat[3*atom_A:3*atom_A+3,3*atom_B:3*atom_B+3]
                for i in range(3):
                    for j in range(3):
                        self.H_AB_vec.extend([H_APF[i,j]])
        return


    def connect_Feature_Target(self):
        #self.features.get_Feature_homonuclear(R_MI_APF_mat=self.R_MI_APF_mat,N_atoms=self.N_atoms)

        self.features.get_Feature_heteronuclear(R_MI_APF_mat=self.R_MI_APF_mat,N_atoms=self.N_atoms,xyz = self.xyz,init_R_MI=self.init_R_MI)
        self.features.get_Feature_homonuclear(R_MI_APF_mat=self.R_MI_APF_mat,N_atoms=self.N_atoms,init_R_MI=self.init_R_MI)

        return self.features.Feature_AB, self.H_AB_vec, self.features.Feature_AA, self.H_AA_vec

    def get_pred_hessian(self,hessian_list):

        self.pred_hess = []
        return

    def get_coord_state(self):
        return self.coord_state[0]


class Feature:
    def __init__(self,folder):
        GFN2_quantities = pd.read_csv(f'{folder}')
        self.CN = np.array(GFN2_quantities.loc[:,['coordination number','delta coordination number']].values.tolist())
        self.dipm_atom = np.array(GFN2_quantities.loc[:,['dipm_atom_x','dipm_atom_y','dipm_atom_z']].values.tolist())
        self.dipm_delta = np.array(GFN2_quantities.loc[:,['dipm_delta_x','dipm_delta_y','dipm_delta_z']].values.tolist())
        self.dipm_only_mull = np.array(GFN2_quantities.loc[:,['delta dipm only mull x','delta dipm only mull y','delta dipm only mull z']].values.tolist())
        self.qm_atom = qm_matrix(np.array(GFN2_quantities.loc[:,['qm_atom_xx','qm_atom_yy', 'qm_atom_zz','qm_atom_xy','qm_atom_zx','qm_atom_yz']].values.tolist()))
        self.qm_delta = qm_matrix(np.array(GFN2_quantities.loc[:,['qm_delta_xx','qm_delta_yy', 'qm_delta_zz','qm_delta_xy','qm_delta_zx','qm_delta_yz']].values.tolist()))
        self.energy_based = np.array(GFN2_quantities.loc[:,['gap (eV)','chem.pot (eV)','HOAO (eV)','LUAO (eV)',
                                    'E_repulsion','E_EHT',' E_disp_2','E_disp_3','E_ies_ixc','E_aes',' E_tot',
                                    'E_axc',' chem_pot_ext','e_gap_ext','ehoao_ext','eluao_ext']].values.tolist())
        self.names = GFN2_quantities.columns.tolist()

        self.Feature_AB = []
        self.Feature_AA = []

    def get_Feature_heteronuclear(self, label=None, R_MI_APF_mat=None, N_atoms = None,xyz = None,init_R_MI=None):

        index = [[2,0,0],
                [1,-1,0],
                [1,0,-1],
                [-1,1,0],
                [0,2,0],
                [0,1,-1],
                [-1,0,1],
                [0,-1,1],
                [0,0,2]]

        for atom_A in range(N_atoms):
            for atom_B in range(atom_A+1,N_atoms):

                i0 = 3*atom_A
                i3 = 3*atom_A + 3
                j0 = 3*atom_B 
                j3 = 3*atom_B + 3

                R_MI_APF = R_MI_APF_mat[i0:i3,j0:j3]

                R_AB = linalg.norm(xyz.iloc[atom_A,1:] -xyz.iloc[atom_B,1:])

                Quantity_AB = [[],[]]

                vector_of_ones = np.array([1,1,1])

                j = 0

                for i in [atom_A,atom_B]:

                    dipm_atom = matmul(init_R_MI,self.dipm_atom[i])
                    dipm_delta = matmul(init_R_MI,self.dipm_delta[i])
                    dipm_only_mull = matmul(init_R_MI,self.dipm_only_mull[i])

                    dipm_atom = matmul(R_MI_APF,dipm_atom)
                    dipm_delta = matmul(R_MI_APF,dipm_delta)
                    dipm_only_mull = matmul(R_MI_APF,dipm_only_mull)

                    qm_atom = matmul(matmul(init_R_MI,self.qm_atom[i]),np.transpose(init_R_MI))
                    qm_delta = matmul(matmul(init_R_MI,self.qm_delta[i]),np.transpose(init_R_MI))

                    qm_atom = matmul(matmul(R_MI_APF,qm_atom),np.transpose(R_MI_APF))
                    qm_delta = matmul(matmul(R_MI_APF,qm_delta),np.transpose(R_MI_APF))
                    

                    if atom_A ==2 and atom_B == 3:
                        print(dipm_atom)
                        print(qm_atom)

                    qm_atom = matmul(qm_atom,vector_of_ones)
                    qm_delta = matmul(qm_delta,vector_of_ones)


                    Quantity_AB[j].extend(dipm_atom)
                    Quantity_AB[j].extend(dipm_delta)
                    Quantity_AB[j].extend(dipm_only_mull)

                    Quantity_AB[j].extend(qm_atom)
                    Quantity_AB[j].extend(qm_delta)
                    Quantity_AB[j].extend(self.energy_based[i])

                    j+=1

                Quantity_AB_arr = np.array(Quantity_AB)

                Feature_Arith = (Quantity_AB_arr[0] + Quantity_AB_arr[1])/2
                Feature_Prod = (Quantity_AB_arr[0] * Quantity_AB_arr[1])
                Feature_AbsDiff = np.abs(Quantity_AB_arr[0] - Quantity_AB_arr[1])

                for i in range(9):
                    Features = []
                    Features.extend(Feature_Arith)
                    Features.extend(Feature_Prod)
                    Features.extend(Feature_AbsDiff)
                    Features.extend([R_AB])
                    Features.extend(index[i])
                    self.Feature_AB.append(Features)
        return





    def get_Feature_homonuclear(self, R_MI_APF_mat=None, N_atoms = None,init_R_MI = None):

        index = [[2,0,0],
                [1,-1,0],
                [1,0,-1],
                [-1,1,0],
                [0,2,0],
                [0,1,-1],
                [-1,0,1],
                [0,-1,1],
                [0,0,2]]


        for atom_A in range(N_atoms):

            i0 = 3*atom_A
            i3 = 3*atom_A + 3

            R_MI_APF = R_MI_APF_mat[i0:i3,i0:i3]

            Quantity_A = []

            vector_of_ones = np.array([1,1,1])
            dipm_atom = matmul(init_R_MI,self.dipm_atom[atom_A])
            dipm_delta = matmul(init_R_MI,self.dipm_delta[atom_A])
            dipm_only_mull = matmul(init_R_MI,self.dipm_only_mull[atom_A])

            dipm_atom = matmul(R_MI_APF,self.dipm_atom[atom_A])
            dipm_delta = matmul(R_MI_APF,self.dipm_delta[atom_A])
            dipm_only_mull = matmul(R_MI_APF,self.dipm_only_mull[atom_A])

            qm_atom = matmul(matmul(R_MI_APF,self.qm_atom[atom_A]),np.transpose(R_MI_APF))
            qm_delta = matmul(matmul(R_MI_APF,self.qm_delta[atom_A]),np.transpose(R_MI_APF))

            qm_atom = matmul(qm_atom,vector_of_ones)
            qm_delta = matmul(qm_delta,vector_of_ones)


            Quantity_A.extend(dipm_atom)
            Quantity_A.extend(dipm_delta)
            Quantity_A.extend(dipm_only_mull)
            Quantity_A.extend(qm_atom)
            Quantity_A.extend(qm_delta)
            Quantity_A.extend(self.energy_based[atom_A])

            for i in range(9):
                Features = []
                Features.extend(Quantity_A)
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
