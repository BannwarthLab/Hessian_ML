import os 
from ReadWrite import ReadWrite
from Rotation_func import Rotation_Functions
from Preparation import Preparation
from HessTarget import HessTarget
from HessFeature import HessFeature

class Geometry(HessTarget,HessFeature, ReadWrite,Preparation,Rotation_Functions):#Feature Class --> picks the right feature; same for target class or: ML Class, picks both

    def __init__(self):
        super().__init__
        return 
        
    def gen_data(self,geo_file,geo,mol):
        self.geo = geo
        self.mol = mol
        self.geo_working_dir = geo_file
        self.xyz,self.header = self.import_coord(os.path.join(self.geo_working_dir,self.file_coord))
        self.dipm = self.import_dipm(os.path.join(self.geo_working_dir,self.file_dipm)).iloc[:,:-3]
        self.hessian = self.import_hessian(os.path.join(self.geo_working_dir,self.file_target),self.xyz)
        self.ml_features = self.import_ml_features(os.path.join(self.geo_working_dir,self.file_feature))

        self.N_atoms = len(self.xyz['atoms'])

        self.init_R_MI,self.xyz = (self.calc_R(self.xyz))

        self.init_P_MI = self.rotM_hess(self.init_R_MI,self.xyz)

        self.rot_init_inert()
        self.rot_inert_apf()

        self.get_Feature_heteronuclear()
        self.get_Feature_homonuclear()

        self.gen_Hessian_vector(self.transpose_list)
        return

    def clear_quantities(self):

        del self.CN
        del self.dipm_atom
        del self.dipm_delta
        del self.dipm_only_mull
        del self.qm_atom
        del self.qm_delta
        del self.energy_based
        del self.hessian
        del self.init_P_MI

        return 

    def get_feature(self):
        return self.Feature_AA, self.Feature_AB

    def get_target(self):
        return self.Target_AA, self.Target_AB


class PickleData:
    def __init__(self,some_class) -> None:

        self.mol = some_class.mol
        self.geo = some_class.geo

        self.xyz = some_class.xyz
        self.N_at = some_class.N_atoms

        self.Feature_AA = some_class.Feature_AA
        self.Feature_AB = some_class.Feature_AB

        self.Target_AA = some_class.Target_AA
        self.Target_AB = some_class.Target_AA

        self.init_R_MI = some_class.init_R_MI
        self.R_MI_APF_mat = some_class.R_MI_APF_mat

        return 

    def add_idx(self,idx):
        self.idx = idx
        return