from hess_ml.src.Parser import Parser
from hess_ml.src.DataGeneration import DataGeneration
from hess_ml.src.Training import Training
from hess_ml.src.Predicting import Predicting
from hess_ml.src.Geometry import Geometry
from hess_ml.src.IO import Input

from sklearn.model_selection import train_test_split

import os 
import numpy as np
import time as time 

class Environment(DataGeneration,Parser,Training,Predicting,Input):
    def __init__(self):
        super().__init__      #initializes all parent classes
        DataGeneration.__init__(self)
        Parser.__init__(self)
        return
    
    def set_general_config(self):

        self.rnd_seed = self.config.get('general',{'random_seed':np.random.randint(0,10000)}).get('random_seed',np.random.randint(0,10000))
        self.threads = self.config.get('threads',1)

        print('Random seed:', self.rnd_seed)

        return 
    
    def import_data(self):

        if self.config.get('general',{'feature':True}).get('feature'): #Maybe adapt these infos in a set_config function

            self.parse_data_set(self.config['geometry'].get('folder'))

            self.do_preparation_split(self.folders,len(self.folders),train_size=self.config['train']['train_size'],
                                    test_size=self.config['train']['test_size'],rnd_seed=self.rnd_seed)
            
            self.gen_features(self.train_idx)    

        return 


    def do_train(self):


        if self.config.get('general',{'train':True}).get('train',True):

            self.train_size = self.config['train']['train_size']

            self.model_name = self.config['train'].get('model_name','ML_Hess')

            self.runtype=self.config.get('runtype','hessian')
            

            if type(self.train_size) == list:

                self.train_size = sorted(self.train_size)[::]

                print(self.train_size)

                for i in range(len(self.train_size)):

                    self.shuffle_idx = np.arange(len(self.Features))

                    print(f'Percentage of data set used for training: {self.train_size[i]*100} %')

                    temp_time_old = time.time()

                    #self.do_train_split(i)
                    train_size = self.train_size[i]/self.train_size[-1]
                    self.shuffle_idx,temp = train_test_split(self.shuffle_idx,train_size=np.clip(train_size,0.0,1.0-1e-8),random_state=self.rnd_seed)
                    
                    del temp 
                    print(self.shuffle_idx)
                    print(f'Total training strucutres:{len(self.shuffle_idx)}')

                    #self.import_FT()

                    self.training_model(i_split=i)

                    temp_time_new = time.time()

                    print(f'Training was done in {round(temp_time_new - temp_time_old)} s' )


                    if self.config['train']['test_size'] > 0.0:
                        
                        temp_time_old = time.time()

                        self.predict(self.test_geo,folder=f'Model{i}/')

                        self.error_estimation(self.test_geo,self.rnd_seed,self.train_size[i])

                        temp_time_new = time.time()

                        print(f'Testing was done in {temp_time_new - temp_time_old: 0.2f} s')

            else:

                print(f'Percentage of data set used for training: {self.train_size*100} %')
                
                self.shuffle_idx = np.arange(len(self.Features))

                temp_time_old = time.time()

                #self.files = self.train_geo

                #self.import_FT()

                self.training_model()

                temp_time_new = time.time()

                print(f'Training was done in {round(temp_time_new - temp_time_old)} s' )


                if self.config['train']['test_size'] > 0.0:

                    temp_time_old = time.time()

                    self.predict(self.test_geo)

                    self.error_estimation(self.test_geo,self.rnd_seed,self.train_size)

                    temp_time_new = time.time()

                    print(f'Testing was done in {temp_time_new - temp_time_old: 0.2f} s')


        return
    
    

    def do_prediction(self):

        if self.config.get('predict',False):

            if self.config['predict'].get('folder',False):
                
                self.parse_data_set(self.config['predict'].get('folder'))

            if self.config['predict'].get('predict_list',False):
                
                files = self.rd_txt_file(self.predict_files)

                try:

                    self.folders.append(files)

                except:

                    self.folders = files

            self.model_name = self.config['predict'].get('model_name','MLHess')

            print(f'Starting prediction of {len(self.folders)} files')

            temp_time_old = time.time()                

            self.predict(self.folders)
    
            temp_time_new = time.time()

            print(f'Prediction was done in {round(temp_time_new - temp_time_old)} s')

        return 

    def gen_features(self,idx):

        if self.config.get('geometry',{'feature':'tblite'}).get('feature'):

            self.Targets = list()
            self.Features = list()
            
            self.generate_data(idx)
            
            self.Targets = np.array(self.Targets).astype(np.float32)
            self.Features = np.array(self.Features).astype(np.float32)

            np.savetxt('Features.txt',self.Features)
            np.savetxt('Targets.txt',self.Targets )

        else:

            if self.feature_import.lower() == 'numpy':

                self.Features = np.loadtxt('Features.txt')
                self.Targets  = np.loadtxt('Targets.txt')
        
        #one could add different features that will be imported
        
        return