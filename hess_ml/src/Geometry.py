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
import time as time
class Geometry(HessTarget,HessFeature, Input,Preparation,Observables,Rotation_Functions):
    
    def __init__(self):
        super().__init__
        return 
        
    def gen_data(self,geo_file,threads):
        
        self.geo_working_dir = geo_file

        self.xyz_pd,self.header = self.import_coord(os.path.join(self.geo_working_dir,self.file_coord))

        self.elements = self.xyz_pd['atoms']

        self.xyz = np.array(self.xyz_pd.iloc[:,1:])

        self.N_atoms = len(self.elements)

        self.nuc_charge = np.zeros(self.N_atoms)

        for i in range(self.N_atoms):
            self.nuc_charge[i] = Const.ELEMENTS2Z[self.elements[i]]


        self.import_ml_features()

        cur_time = time.time()
        
        self.hessian = self.import_hessian(os.path.join(self.geo_working_dir,self.file_target),self.N_atoms)

        self.filter_feature()

        self.import_gradient(os.path.join(self.geo_working_dir,self.file_gradient))
        print('Importing procedure:',round(time.time()- cur_time,5),'s')

        cur_time = time.time()
        self.rot_inert_apf()
        print('Rotation procedure:',round(time.time()- cur_time,5),'s')

        cur_time = time.time()
        self.get_Feature(threads)
        print('Feature:',round(time.time()- cur_time,5),'s')

        cur_time = time.time()

        self.gen_Hessian_vector(self.transpose_list)
        print('Hessian:',round(time.time()- cur_time,5),'s')

        print(np.array(self.Feature_AB).shape,np.array(self.Target_AB).shape)
        
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
