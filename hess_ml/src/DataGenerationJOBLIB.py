import os 

import numpy as np

import time as time

from sklearn.model_selection import train_test_split

from hess_ml.src.Geometry import Geometry
from hess_ml.src.SaveDat import FTHetero,FTHomo,PickleData,PickleDict

from joblib import Parallel, delayed

from multiprocessing import current_process
import pickle as pickle
import glob as glob


class DataGeneration(Geometry):
    
    def __init__(self) -> None:
        Geometry.__init__
        return
                
    def generate_feature_target_sf_dtr(self):
        self.feature_target_file = ['Feature_Vector_Homo','Target_Vector_Homo','Feature_Vector_Hetero','Target_Vector_Hetero']
        
        for file in ['Model_Homo*.json','Model_Hetero*.json','test_structures*.json',self.output_file]:
            files = glob.glob(file)
            for f in files:
                os.remove(f)
            #self.truncate_file(file)

        print(f'Generating Features from {self.folder_data}')

        if self.train_test == True:
            total_structures = 0
            self.geo_dir = []
            molecule_dir = sorted([mol for mol in os.listdir(f'{self.cwd}/{self.folder_data}') if os.path.isdir(os.path.join(f'{self.cwd}/{self.folder_data}',mol))])
            for mol in range(len(molecule_dir)):
                data_dir = os.path.join(f'{self.cwd}/{self.folder_data}',molecule_dir[mol])
                temp_dir = sorted([os.path.join(data_dir,geo) for geo in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,geo))])
                self.geo_dir.extend(temp_dir)
                total_structures += len(temp_dir)

            geo_idx = np.arange(0,total_structures-1)
            print(f'{self.train_size*100} % of the set is used for training.')
            print(f'{self.test_size*100} % of the set is used for testing.')

            self.train_idx, self.test_idx  = train_test_split(geo_idx,test_size=self.test_size,train_size=self.train_size,random_state=self.rnd_seed)

            np.savetxt('test_idx.txt',self.test_idx)
            np.savetxt('train_idx.txt',self.train_idx)

            self.comp_idx = np.concatenate((self.train_idx,self.test_idx),axis=None)
            #self.geo_dir = np.array(self.geo_dir)[self.comp_idx]
            geo_idx = geo_idx[self.comp_idx]
        else:
            molecule_dir = sorted([mol for mol in os.listdir(f'{self.cwd}/{self.folder_data}') if os.path.isdir(os.path.join(f'{self.cwd}/{self.folder_data}',mol))])

        self.wall_time0 = time.time()

        self.idx_list = []
        
        #________Paralleized Feature Generation___________ 
        Parallel(n_jobs=self.threads)(delayed(self.generation_procedure_dtr)(geom=geo,mol=mol,dir=self.geo_dir[geo]) for geo in self.comp_idx)

        print(f'Features and Targets of {len(self.train_idx)+len(self.test_idx)} structures were generated in {round(time.time() - self.wall_time0)} s\n')

        return


    def generation_procedure_dtr(self,mol=None,geom=None,dir=None):
        #INIT GEOMETRY
        #GENERATE TARGET & FEATURE --> picks dependend on env the right features and target

        if geom in self.comp_idx:

            self.gen_data(self.geo_dir[geom],mol,geom)
            self.clear_quantities()
            idx = [self.mol,geom]

            if geom in self.train_idx:

                het = FTHetero(self,mol,geom,dir)

                with open(f'Model_Hetero{current_process()._identity[0]}.json','ab+') as g:
                    pickle.dump(het.dict,g)

            if geom in self.test_idx:

                struc = PickleData(self,mol,geom,dir)
                with open(f'test_structures{current_process()._identity[0]}.json','ab+') as h:
                    pickle.dump(struc.dict,h)

            self.idx_list.append(idx)

        return


    def truncate_file(self,file):

        if os.path.isfile(file):
            with open(file,'wb') as f1:
                f1.truncate(0)
        
            f1.close()

        return 

