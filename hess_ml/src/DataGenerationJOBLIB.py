import os 

import numpy as np

import time as time

from sklearn.model_selection import train_test_split

from src.Geometry import Geometry
from src.SaveDat import FTHetero,FTHomo,PickleData

from joblib import Parallel, delayed

from multiprocessing import current_process
import pickle as pickle
import glob as glob


class DataGeneration(Geometry):
    
    def __init__(self) -> None:
        Geometry.__init__
        return
                
    def generate_feature_target_sf(self):
        self.feature_target_file = ['Feature_Vector_Homo','Target_Vector_Homo','Feature_Vector_Hetero','Target_Vector_Hetero']
        
        for i in range(1,self.threads+1):
            if os.path.isfile(f'Model_Homo{i}.json'):
                with open(f'Model_Homo{i}.json','wb') as f1:
                    f1.truncate(0)

            if os.path.isfile(f'Model_Hetero{i}.json'):
                with open(f'Model_Hetero{i}.json','wb') as f2:
                    f2.truncate(0)

            if os.path.isfile(f'test_structures{i}.json'):
                with open(f'test_structures{i}.json','wb') as f3:
                    f3.truncate(0)

            if os.path.isfile(self.output_file):
                with open(self.output_file,'wb') as f4:
                    f4.truncate(0)

        print(f'Generating Features from {self.folder_data}')

           
        if self.train_test == True:
            total_structures = 0
            molecule_dir = sorted([mol for mol in os.listdir(f'{self.cwd}/{self.folder_data}') if os.path.isdir(os.path.join(f'{self.cwd}/{self.folder_data}',mol))])
            for mol in range(len(molecule_dir)):
                data_dir = os.path.join(f'{self.cwd}/{self.folder_data}',molecule_dir[mol])
                geo_dir = sorted([geo for geo in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,geo))])
                total_structures +=len(sorted([geo for geo in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,geo))]))

            geo_idx = np.arange(total_structures)

            print(f'{self.train_size*100} % of the set is used for testing.')
            print(f'{self.test_size*100} % of the set is used for training.')

            self.train_idx, self.test_idx  = train_test_split(geo_idx,test_size=self.test_size,train_size=self.train_size,random_state=self.rnd_seed)
            np.savetxt('test_idx.txt',self.test_idx)
            np.savetxt('train_idx.txt',self.train_idx)
            self.comp_idx = self.train_idx.copy().extend(self.test_idx.copy())

        #_______Reading_the_names_of_all_folders______
        self.wall_time0 = time.time()

        self.count = 0
        self.idx_list = []
        molecule_dir = sorted([mol for mol in os.listdir(f'{self.cwd}/{self.folder_data}') if os.path.isdir(os.path.join(f'{self.cwd}/{self.folder_data}',mol))])
        for mol in range(len(molecule_dir)):
            #_______Reading_the_names_of_all_subfolders_______
            self.data_dir = os.path.join(f'{self.cwd}/{self.folder_data}',molecule_dir[mol])

            self.geo_dir = sorted([geo for geo in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,geo))])

            #self.rest_array = np.arange(len(geo_dir),len(geo_dir)+(self.threads-1)-len(geo_dir)%(self.threads-1))

            #for geo in range(len(geo_dir)):
                #_____Rotating_Feature_and_Target_for_each_system_____
                #_____MPI_parallelized_with_mpi4py____________________
            # self.het_files = []
            # self.hom_files = []
            # self.test_files = []

            #for i in range(self.threads):
            #self.het_files.append(tf.TemporaryFile(mode='w+b',suffix='.json',prefix='Other',dir=self.cwd))
                #self.hom_files.append(tf.NamedTemporaryFile(mode='w+b',suffix='.json',prefix='HomoModel',dir=self.cwd))
                #self.test_files.append(tf.NamedTemporaryFile(mode='w+b',suffix='.json',prefix='test_structures',dir=self.cwd))

            dir_idx = np.arange(len(geo_dir))
            #with parallel_backend('loky',n_jobs=self.threads):
            Parallel(n_jobs=self.threads)(delayed(self.generation_procedure)(geom=geo,mol=mol) for geo in dir_idx)
            #            with parallel_backend('loky',n_jobs=self.threads):
            #for geo in dir_idx:
            #    self.generation_procedure(geom=geo,mol=mol) #for geo in dir_idx

            print(f'Features and Targets of {len(geo_dir)} structures were generated in {round(time.time() - self.wall_time0)} s\n')
                
        return


    def generation_procedure(self,mol=None,geom=None):
        #INIT GEOMETRY
        #
        #GENERATE TARGET & FEATURE --> picks dependend on env the right features and target
        if geom in self.comp_idx:

            self.gen_data(os.path.join(self.data_dir,self.geo_dir[geom]),mol,geom)

            self.clear_quantities()
            idx = [self.mol,geom]

            if geom in self.train_idx:

                het = FTHetero(self,mol,geom)
                hom = FTHomo(self,mol,geom)

                with open(f'Model_Homo{current_process()._identity[0]}.json','ab+') as f:
                    pickle.dump(hom,f)

                with open(f'Model_Hetero{current_process()._identity[0]}.json','ab+') as g:
                    pickle.dump(het,g)

            if geom in self.test_idx:

                struc = PickleData(self,mol,geom)

                for i in range(len(struc.Feature_AB)):
                    if len(struc.Feature_AB[i]) != 169:
                        print(struc.geo,i)

                with open(f'test_structures{current_process()._identity[0]}.json','ab+') as h:
                    pickle.dump(struc,h)

            self.idx_list.append(idx)
        return