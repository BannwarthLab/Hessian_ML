import pandas as pd
from rotation_func import *

class sys_info:
    def __init__(self,folder,molecule,variation):
        self.xyz, self.header = import_coord(f'{folder}coord.xyz')
        self.hessian = import_hessian(f'{folder}/hessian',self.xyz)
        self.dipm = import_dipm(f'{folder}/xyz_dipm.csv').iloc[:,:-3]
        self.GFN2_qauntities = pd.read_csv(f'{folder}/ml_feature.csv')
        self.init_R_MI = np.transpose(calc_R(self.xyz))
        self.init_P_MI = np.transpose(rotM_hess(self.init_R_MI,self.xyz))
        self.coord_state = ['init','inert']
        self.molecule = molecule
        self.variation = variation
        self.pred_hess = None

    def rot_init_inert(self):
        self.coord_state[0], self.coord_state[1] = self.coord_state[1],self.coord_state[0]
        self.init_R_MI = np.transpose(self.init_R_MI)
        self.init_P_MI = np.transpose(self.init_P_MI)
        self.xyz = coord_rot(self.xyz,self.init_R_MI)
        self.dipm = coord_rot(self.dipm,self.init_R_MI)
        self.hessian = matmul(matmul(self.init_P_MI,self.hessian),np.transpose(self.init_P_MI))
        return 

    def get_Feature_Target(self, label = 'permuted'):

        a = self.GFN2_qauntities

        if label =='permuted':
            print('Labeling: Permutation')

        elif label =='indexed':
            print('Labeling: Indexing')

        return 

    def get_pred_hessian(self,hessian_list):

        self.pred_hess = []
        return

    def get_coord_state(self):
        return self.coord_state[0]


class Feature:
    def __init__(self,folder):
        GFN2_quantities = pd.read_csv(f'{folder}/ml_feature.csv')
        self.CN = np.array(GFN2_quantities.loc[:,['coordination number','delta coordination number']].values.tolist())
        self.dipm_atom = np.array(GFN2_quantities.loc[:,['dipm_atom_x','dipm_atom_y','dipm_atom_z']].values.tolist())
        self.dipm_delta = np.array(GFN2_quantities.loc[:,['dipm_delta_x','dipm_delta_y','dipm_delta_z']].values.tolist())
        self.dipm_only_mull = np.array(GFN2_quantities.loc[:,['delta dipm only mull x','delta dipm only mull y','delta dipm only mull z']].values.tolist())
        self.qm_atom = qm_matrix(np.array(GFN2_quantities.loc[:,['qm_atom_xx','qm_atom_yy', 'qm_atom_zz','qm_atom_xy','qm_atom_zx','qm_atom_yz']].values.tolist()))
        self.qm_delta = qm_matrix(np.array(GFN2_quantities.loc[:,['qm_delta_xx','qm_delta_yy', 'qm_delta_zz','qm_delta_xy','qm_delta_zx','qm_delta_yz']].values.tolist()))
        self.energy_based = np.array(GFN2_quantities.loc[:,['response (a.u.)','gap (eV)','chem.pot (eV)','HOAO (eV)','LUAO (eV)',
                                    'E_repulsion','E_EHT',' E_disp_2','E_disp_3','E_ies_ixc','E_aes',' E_tot',
                                    'E_axc',' chem_pot_ext','e_gap_ext','ehoao_ext','eluao_ext']].values.tolist())
        self.names = GFN2_quantities.columns.tolist()

    def feature(self,label='indexed'):
        return


    def get_Feature(self,label=None,R_MI_APF_l=None):
        features = []
        self.CN[:,[A,B]]
        return 



def qm_matrix(qm_atom):
    qm_matrix_list = []
    for i in range(len(qm_atom)):
        xx = qm_atom[i,0]
        xy = qm_atom[i,1]
        yy = qm_atom[i,2]
        xz = qm_atom[i,3]
        zz = qm_atom[i,4]
        yz = qm_atom[i,5]

        qm_matrix = np.array([[xx,xy,xz],
                        [xy,yy,yz],
                        [xz,yz,zz]])

        qm_matrix_list.append(qm_matrix)

    return qm_matrix_list
