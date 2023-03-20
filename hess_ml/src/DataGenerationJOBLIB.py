import os 

import numpy as np

import time as time

from sklearn.model_selection import train_test_split

from src.Geometry import Geometry
from src.SaveDat import FTHetero,FTHomo,PickleData,PickleDict

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
            molecule_dir = sorted([mol for mol in os.listdir(f'{self.cwd}/{self.folder_data}') if os.path.isdir(os.path.join(f'{self.cwd}/{self.folder_data}',mol))])
            for mol in range(len(molecule_dir)):
                data_dir = os.path.join(f'{self.cwd}/{self.folder_data}',molecule_dir[mol])
                geo_dir = sorted([geo for geo in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,geo))])
                total_structures +=len(sorted([geo for geo in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,geo))]))

            geo_idx = np.arange(total_structures)

            print(f'{self.train_size*100} % of the set is used for training.')
            print(f'{self.test_size*100} % of the set is used for testing.')

            self.train_idx, self.test_idx  = train_test_split(geo_idx,test_size=self.test_size,train_size=self.train_size,random_state=self.rnd_seed)
            np.savetxt('test_idx.txt',self.test_idx)
            np.savetxt('train_idx.txt',self.train_idx)
            self.comp_idx = np.concatenate((self.train_idx,self.test_idx),axis=None)
            geo_dir = np.array(geo_dir)[self.comp_idx]
        else:
            molecule_dir = sorted([mol for mol in os.listdir(f'{self.cwd}/{self.folder_data}') if os.path.isdir(os.path.join(f'{self.cwd}/{self.folder_data}',mol))])

        #_______Reading_the_names_of_all_folders______
        self.wall_time0 = time.time()

        self.count = 0
        self.idx_list = []
        for mol in range(len(molecule_dir)):
            #_______Reading_the_names_of_all_subfolders_______ 
            self.data_dir = os.path.join(f'{self.cwd}/{self.folder_data}',molecule_dir[mol])

            self.geo_dir = sorted([geo for geo in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,geo))])
            dir_idx = np.arange(len(geo_dir))
            #________Paralleized Feature Generation___________ 
            Parallel(n_jobs=self.threads)(delayed(self.generation_procedure_dtr)(geom=geo,mol=mol,dir=geo_dir[geo]) for geo in dir_idx)

            print(f'Features and Targets of {len(self.train_idx)+len(self.test_idx)} structures were generated in {round(time.time() - self.wall_time0)} s\n')

        return


    def generation_procedure_dtr(self,mol=None,geom=None,dir=None):
        #INIT GEOMETRY
        #
        #GENERATE TARGET & FEATURE --> picks dependend on env the right features and target
        if geom in self.comp_idx:

            self.gen_data(os.path.join(self.data_dir,self.geo_dir[geom]),mol,geom)

            self.clear_quantities()
            idx = [self.mol,geom]

            if geom in self.train_idx:

                het = FTHetero(self,mol,geom,dir)
                hom = FTHomo(self,mol,geom,dir)

                with open(f'Model_Homo{current_process()._identity[0]}.json','ab+') as f:
                    pickle.dump(hom.dict,f)

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


    def generate_feature_target_sf_gnn(self):
        self.feature_target_file = ['Feature_Vector_Homo','Target_Vector_Homo','Feature_Vector_Hetero','Target_Vector_Hetero']
        
        gnn_strucs = glob.glob('GNN_structures*.json')
        for file in gnn_strucs:
            self.truncate_file(file)

        print(f'Generating Features from {self.folder_data}')

           
        total_structures = 0
        molecule_dir = sorted([mol for mol in os.listdir(f'{self.cwd}/{self.folder_data}') if os.path.isdir(os.path.join(f'{self.cwd}/{self.folder_data}',mol))])
        for mol in range(len(molecule_dir)):
            data_dir = os.path.join(f'{self.cwd}/{self.folder_data}',molecule_dir[mol])
            geo_dir = sorted([geo for geo in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,geo))])
            total_structures +=len(sorted([geo for geo in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,geo))]))
       
        if self.train_test == True:

            geo_idx = np.arange(total_structures)

            print(f'{self.train_size*100} % of the set is used for testing.')
            print(f'{self.test_size*100} % of the set is used for training.')

            self.train_idx, self.test_idx  = train_test_split(geo_idx,test_size=self.test_size,train_size=self.train_size,random_state=self.rnd_seed)
            np.savetxt('test_idx.txt',self.test_idx)
            np.savetxt('train_idx.txt',self.train_idx)
            self.comp_idx = np.concatenate((self.train_idx,self.test_idx),axis=None)

        #_______Reading_the_names_of_all_folders______
        self.wall_time0 = time.time()

        self.count = 0
        self.idx_list = []
        molecule_dir = sorted([mol for mol in os.listdir(f'{self.cwd}/{self.folder_data}') if os.path.isdir(os.path.join(f'{self.cwd}/{self.folder_data}',mol))])
        for mol in range(len(molecule_dir)):
            #_______Reading_the_names_of_all_subfolders_______
            self.data_dir = os.path.join(f'{self.cwd}/{self.folder_data}',molecule_dir[mol])

            self.geo_dir = sorted([geo for geo in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,geo))])
            dir_idx = np.arange(len(geo_dir))

            #________Paralleized Feature Generation___________ 
            Parallel(n_jobs=self.threads)(delayed(self.generation_procedure_gnn)(geom=geo,mol=mol,dir=geo_dir[geo]) for geo in dir_idx)

            print(f'Features and Targets of {len(geo_dir)} structures were generated in {round(time.time() - self.wall_time0)} s\n')
                
        return


    def generation_procedure_gnn(self,mol=None,geom=None,dir =None):
        #INIT GEOMETRY
        #
        #GENERATE TARGET & FEATURE --> picks dependend on env the right features and target

        self.gen_data(os.path.join(self.data_dir,self.geo_dir[geom]),mol,geom)

        struc = PickleDict(self,mol,geom,dir)
        
        with open(f'GNN_structures{current_process()._identity[0]}.json','ab+') as h:
            pickle.dump(struc.dict,h)

        return
