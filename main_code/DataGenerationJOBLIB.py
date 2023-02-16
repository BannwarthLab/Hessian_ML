import os 

import numpy as np

import time as time

from sklearn.model_selection import train_test_split

from Geometry import Geometry
from SaveDat import FTHetero,FTHomo
from Geometry import PickleData
from joblib import Parallel, delayed, parallel_backend
import multiprocessing as mp 
import pickle as pickle
class DataGeneration(Geometry):
    
    def __init__(self) -> None:
        Geometry.__init__
        return
                
    def generate_feature_target_sf(self):
        self.feature_target_file = ['Feature_Vector_Homo','Target_Vector_Homo','Feature_Vector_Hetero','Target_Vector_Hetero']
        

        if os.path.isfile('Model_Homo'):
            with open(f'Model_Homo.json','wb') as f:
                f.truncate(0)

        if os.path.isfile('Model_Hetero'):
            with open(f'Model_Hetero.json','wb') as f:
                f.truncate(0)

        if os.path.isfile(self.output_file):
            with open(self.output_file,'wb') as f:
                f.truncate(0)

        print(f'Generating Features from {self.folder_data}')

           
        if self.train_test == True:
            total_structures = 0
            molecule_dir = sorted([mol for mol in os.listdir(f'{self.cwd}/{self.folder_data}') if os.path.isdir(os.path.join(f'{self.cwd}/{self.folder_data}',mol))])
            for mol in range(len(molecule_dir)):
                data_dir = os.path.join(f'{self.cwd}/{self.folder_data}',molecule_dir[mol])
                geo_dir = sorted([geo for geo in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,geo))])
                total_structures +=len(sorted([geo for geo in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,geo))]))

            geo_idx = np.arange(total_structures)

            self.train_idx, self.test_idx  = train_test_split(geo_idx,test_size=self.test_size,train_size=self.train_size,random_state=self.rnd_seed)
            np.savetxt('test_idx.txt',self.test_idx)
            np.savetxt('train_idx.txt',self.train_idx)

        #_______Reading_the_names_of_all_folders______
        self.wall_time0 = time.time()

        self.count = 0
        self.idx_list = []
        print(len(self.train_idx))
        molecule_dir = sorted([mol for mol in os.listdir(f'{self.cwd}/{self.folder_data}') if os.path.isdir(os.path.join(f'{self.cwd}/{self.folder_data}',mol))])
        for mol in range(len(molecule_dir)):
            #_______Reading_the_names_of_all_subfolders_______
            self.data_dir = os.path.join(f'{self.cwd}/{self.folder_data}',molecule_dir[mol])

            self.geo_dir = sorted([geo for geo in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,geo))])

            #self.rest_array = np.arange(len(geo_dir),len(geo_dir)+(self.threads-1)-len(geo_dir)%(self.threads-1))

            #for geo in range(len(geo_dir)):
                #_____Rotating_Feature_and_Target_for_each_system_____
                #_____MPI_parallelized_with_mpi4py____________________
            dir_idx = np.arange(len(geo_dir))

            print(len(geo_dir))

            with parallel_backend('loky',n_jobs=self.threads):
                Parallel()(delayed(self.generation_procedure)(geom=geo,mol=mol) for geo in dir_idx)

            print(f'A total of {len(geo_dir)} structures were imported in {round(time.time() - self.wall_time0)} s')
                
        return


    def generation_procedure(self,mol=None,geom=None):

        #INIT GEOMETRY
        #
        #GENERATE TARGET & FEATURE --> picks dependend on env the right features and target
        self.gen_data(os.path.join(self.data_dir,self.geo_dir[geom]),mol,geom)

        het = FTHetero(self)
        hom = FTHomo(self)

        self.clear_quantities()
        idx = [self.mol,geom]

        if geom in self.train_idx:
            with open(f'Model_Homo.json','ab+') as f:
                pickle.dump(hom,f)
            #f.close()

            with open(f'Model_Hetero.json','ab+') as g:
                pickle.dump(het,g)
            #f.close()

        self.idx_list.append(idx)
        return