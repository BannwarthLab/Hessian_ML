import os 

import numpy as np

import time as time

from sklearn.model_selection import train_test_split

from hess_ml.src.Geometry import Geometry
from hess_ml.src.SaveDat import PickleData
from joblib import Parallel, delayed
import pickle as pickle
import glob as glob


class DataGeneration(Geometry):
    
    def __init__(self) -> None:
        Geometry.__init__
        return
                
    def generate_data(self,idx=None):
        
        self.wall_time0 = time.time()

        print(f'Starting Data Generation for Features...', end="")

        #________Parallelized Feature Generation___________

        #if idx == None:
        #    idx = np.arange(0,len(self.geo_dir))
        
        Parallel(n_jobs=self.threads)(delayed(self.generation_procedure)(dir=self.geo_dir[geo]) for geo in idx)

        print('done')

        print(f'Features and Targets of {len(idx)} structures were generated in {round(time.time() - self.wall_time0)} s\n')

        return


    def generation_procedure(self,dir=None):
        
        self.gen_data(dir)

        self.clear_quantities()

        data = PickleData(self,dir)

        with open(os.path.join(dir,f'Hessian_ML_Data.json'),'ab+') as h:

            pickle.dump(data.dict,h)

        return


    def truncate_file(self,file):

        if os.path.isfile(file):

            with open(file,'wb') as f1:

                f1.truncate(0)
        
            f1.close()

        return


    def parse_folders(self,folder,subfolder):

        print(f'Gathering Folders from {folder}.')

        if subfolder:

            self.gather_subfolders(folder)

        else:
            
            self.gather_folders(folder)

        return 
    
    def gather_subfolders(self,folder):

        self.total_structures = 0

        self.geo_dir = []

        molecule_dir = sorted([mol for mol in os.listdir(f'{folder}') if os.path.isdir(os.path.join(f'{folder}',mol))])

        for mol in range(len(molecule_dir)):
            
            data_dir = os.path.join(f'{folder}',molecule_dir[mol])

            temp_dir = sorted([os.path.join(data_dir,geo) for geo in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,geo))])

            self.geo_dir.extend(temp_dir)
            
            self.total_structures += len(temp_dir)

        return
    

    def gather_folders(self,folder):

        molecule_dir = sorted([mol for mol in os.listdir(f'{folder}') if os.path.isdir(os.path.join(f'{folder}',mol))])

        self.geo_dir = (molecule_dir)

        self.total_structures = len(self.geo_dir)

        return 
    
    def do_preparation_split(self):

        """
        Does a split of the geometry file directories into train and test sets.
        Saves the information in txt files
        """
        
        geo_idx = np.arange(0,self.total_structures-1)

        if type(self.train_size) == list:
            self.train_size_temp = max(self.train_size)
        else:
            self.train_size_temp = self.train_size
        

        self.train_idx, self.test_idx  = train_test_split(geo_idx,test_size=self.test_size,train_size=self.train_size_temp,random_state=self.rnd_seed)

        self.comp_idx = np.concatenate((self.train_idx,self.test_idx),axis=None)

        geo_idx = geo_idx[self.comp_idx]

        self.test_geo = []

        for i in self.test_idx:

            self.test_geo.append(self.geo_dir[i])
        
        self.train_geo = []

        for i in self.train_idx:

            self.train_geo.append(self.geo_dir[i])


        self.data_to_txt(self.test_geo,os.path.join('','test_files.txt'))

        self.data_to_txt(self.train_geo,os.path.join('','train_files.txt'))

        return