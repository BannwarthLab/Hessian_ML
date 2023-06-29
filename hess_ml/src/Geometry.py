import os 
from hess_ml.src.IO import Input

from hess_ml.src.Rotation_func import Rotation_Functions
from hess_ml.src.Preparation import Preparation

from hess_ml.src.HessTarget import HessTarget
from hess_ml.src.HessFeature import HessFeature

from hess_ml.src.Observables import Observables
from hess_ml.src.constants import Const

import json as json
import numpy as np

class Geometry(HessTarget,HessFeature, Input,Preparation,Observables,Rotation_Functions):
    
    def __init__(self):
        super().__init__
        return 
        
    def gen_data(self,geo_file):
        
        self.geo_working_dir = geo_file

        self.xyz,self.header = self.import_coord(os.path.join(self.geo_working_dir,self.file_coord))
        self.N_atoms = len(self.xyz['atoms'])

        self.nuc_charge = np.zeros(self.N_atoms)

        for i in range(self.N_atoms):
            self.nuc_charge[i] = Const.ELEMENTS2Z[self.xyz.loc[i,'atoms']]


        self.hessian = self.import_hessian(os.path.join(self.geo_working_dir,self.file_target),self.xyz)

        self.import_ml_features(os.path.join(self.geo_working_dir,self.file_feature))

        self.filter_feature()

        self.import_gradient(os.path.join(self.geo_working_dir,self.file_gradient))
        
        self.rot_inert_apf()

        self.get_Feature()

        self.gen_Hessian_vector(self.transpose_list)
        
        return

    def clear_quantities(self):

        del self.cn
        del self.p
        del self.q
        del self.dipm
        del self.qm
        del self.energy_based
        del self.hessian

        return 

    def get_feature(self):
        return self.Feature_AA, self.Feature_AB

    def get_target(self):
        return self.Target_AA, self.Target_AB
