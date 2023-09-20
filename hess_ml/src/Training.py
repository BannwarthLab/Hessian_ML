
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import TruncatedSVD
from joblib import dump
from joblib import parallel_backend

import json as json 

from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from hess_ml.src.IO import Input
import os
import numpy as np
import glob as glob
import time as time 

class Training(Input):
    def __init__(self) -> None:
        super().__init__
        pass

    def train(self,runtype):

        self.mode = runtype

        print(self.train_size)

        if type(self.train_size) == list:

            for i in range(len(self.train_size)):

                temp_time_old = time.time()

                self.do_train_test_split(i)

                #self.import_FT()

                self.training_model(i=i)

                temp_time_new = time.time()

                print(f'Training was done in {round(temp_time_new - temp_time_old)} s' )

        else:

            temp_time_old = time.time()


            #self.files = self.train_geo

            #self.import_FT()

            self.training_model()

            temp_time_new = time.time()

            print(f'Training was done in {round(temp_time_new - temp_time_old)} s' )


        return

    def import_FT_old(self):
        
        self.Features = []
        self.Targets = []

        if self.mode == 'hetero':

            print(f'Importing Features and Targets of heteronuclear model... ', end="")

            self.files = glob.glob('TrainData_*.json')

            for f in self.files:
                Feature_temp,Targets_temp = self.import_pickle_FT(f)
                self.Features.extend(Feature_temp)
                self.Targets.extend(Targets_temp)
            self.Features = np.array(self.Features)
            self.Targets = np.array(self.Targets)
            
            print('done\n')

        else:
            print('no feature or target imported')


        return 

    def import_FT(self):

        self.Features = []
        self.Targets = []

        print(f'Importing Features and Targets of hessian model... ', end="")

        for f in self.files:

            Feature_temp,Targets_temp = self.import_pickle_FT(os.path.join(f,f'Hessian_ML_Data.json'))

            self.Features.extend(Feature_temp)

            self.Targets.extend(Targets_temp)

        self.Features = np.array(self.Features)

        self.Targets = np.array(self.Targets)
        
        print('done\n')

        return 


    def training_model(self,i_split = None):

        params = self.config['training']['parameter']


        self.Targets = np.array(self.Targets).astype(np.float32)

        self.Features = np.array(self.Features).astype(np.float32)

        print(f'Feature matrix shape {self.Features.shape}')
        
        print(f'Target matrix shape {self.Targets.shape}')

        if type(params) == list:

            params = params[0]

        method = self.method.lower()
        print(f'Chosen method: {self.method}')

        SearchCV = self.SearchCV
        
        search = False


        if method == 'etr':

            regr_model = ExtraTreesRegressor(random_state=self.rnd_seed)

            regr_model.set_params(**params)

            self.normalization = False

            self.selection = False


        if method == 'rfr':

            regr_model = RandomForestRegressor(random_state=self.rnd_seed)

            regr_model.set_params(**params)

            self.normalization = False


        if method == 'svr':

            single_regr_model = SVR()

            single_regr_model.set_params(**params)

            regr_model = MultiOutputRegressor(single_regr_model)

            self.normalization = True


        if method == 'mlpr':

            regr_model = MLPRegressor()

            regr_model.set_params(**params)

            #regr_model.set_params(hidden_layer_sizes=(210,210,210,210,210,210,210,210,210,210,210,210,210,210,120,30,21))

            self.normalization = True


        if SearchCV.lower() == 'random':

            search = True

            regr_model = RandomizedSearchCV(regr_model,param_distributions=params,n_iter=self.n_iter_search,random_state=self.rnd_seed)


        if SearchCV.lower() == 'grid':

            search = True

            regr_model = GridSearchCV(regr_model,param_grid=params)


        print('Parameters for the Model:')  
        param_temp = regr_model.get_params()
        for param in param_temp.keys():
            print(f'{param}: {param_temp[param]}')


        del param_temp

       


        if self.normalization == True:
            transformer = StandardScaler().fit(self.Features)
            self.Features = transformer.transform(self.Features)

            with open(f'{self.model_name}_transformer.joblib','w') as g:
                g.truncate(0)
            g.close()

            pathname = f'{self.model_name}_transformer.joblib'

            dump(transformer,pathname)
            
            print(f'Transformer for Features is saved in {pathname}.\n')

            target_transformer = StandardScaler().fit(self.Targets)

            self.Targets = target_transformer.transform(self.Targets)

            with open(f'{self.model_name}_transformer_target.joblib','w') as g:
                g.truncate(0)
            g.close()

            pathname = f'{self.model_name}_transformer_target.joblib'

            dump(target_transformer,pathname)
            
            print(f'Transformer for Targets is saved in {pathname}.\n')

        if self.selection:

            #selector = VarianceThreshold(threshold=0.05*(1-0.05))
            
            selector = TruncatedSVD(n_components=200,algorithm='arpack')

            self.Features = selector.fit_transform(self.Features)

            with open(f'{self.model_name}_selector.joblib','w') as g:
                g.truncate(0)
            g.close()

            pathname = f'{self.model_name}_selector.joblib'

            dump(selector,pathname)

            print(f'Selector is saved in {pathname}.\n')

            print(f'Feature vector reduced to shape {self.Features.shape}')
            

        if method != 'mlpr':
            regr_model.set_params(n_jobs=self.threads)
            regr_model.fit(self.Features,self.Targets)

        else:
            with parallel_backend('threading', n_jobs=self.threads):
                regr_model.fit(self.Features,self.Targets)

        print(f'Score on training data: {regr_model.score(self.Features,self.Targets)}')

        if search:
            print(f'Used Parameters:\n {regr_model.best_params_}')

        else:
            print(f'Used Parameters:\n {regr_model.get_params()}')

        with open(f'{self.model_name}.joblib','w') as g:
            g.truncate(0)
        g.close()

        if not(i_split == None):
            pathname = f'Model{i_split}/{self.model_name}.joblib'
        else:
            pathname = f'{self.model_name}.joblib'

        dump(regr_model,pathname)
        print(f'Model is saved in {pathname}.\n')


        del g
        del regr_model
        
        del self.Features
        del self.Targets

        if self.normalization:
            del transformer
            del target_transformer

        if self.selection:
            del selector

        return
    

    def do_train_test_split(self,i):

        """
        Does a split of the geometry file directories into train and test sets.
        Saves the information in txt files
        """

        print(f'{self.train_size[i]*100} % of the set is used for training.')

        self.files, temp  = train_test_split(self.train_geo,train_size=self.train_size[i],test_size=1-self.train_size[i],random_state=self.rnd_seed)

        self.comp_idx = np.concatenate((self.train_idx,self.test_idx),axis=None)

        mypath = f'Model{i}'

        if not os.path.isdir(mypath):

            os.makedirs(mypath)

        self.data_to_txt(self.train_geo,os.path.join(f'Model{i}/','train_files.txt'))

        return