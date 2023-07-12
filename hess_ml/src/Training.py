
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import TruncatedSVD
from joblib import dump

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

    def train(self,train_conf,runtype):

        self.mode = runtype

        print(self.train_size)

        if type(self.train_size) == list:

            for i in range(len(self.train_size)):

                temp_time_old = time.time()

                self.do_train_test_split(i)

                #self.import_FT()

                self.training_model(train_conf,i=i)

                temp_time_new = time.time()

                print(f'Training was done in {round(temp_time_new - temp_time_old)} s' )

        else:

            temp_time_old = time.time()

            self.files = self.train_geo

            #self.import_FT()

            self.training_model(train_conf)

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


    def training_model(self,train_conf,i_split = None):

        params = self.config['training']['parameter']

        self.Features = list()
        self.Targets = list()

        for i in range(len(self.FT)):
            self.Features.extend(self.FT[i][0])
            self.Targets.extend(self.FT[i][1])

        self.Targets = np.array(self.Targets).astype(np.float32)

        self.Features = np.array(self.Features).astype(np.float32)

        print(f'Feature matrix shape {self.Features.shape}')
        print(f'Target matrix shape {self.Targets.shape}')


        if type(params) == list:

            params = params[0]

        print('Parameters for the Model:')  

        for param in params.keys():
            print(f'{param}: {params[param]}')

        print(f'Training hessian ML model with a feature matrix of shape {np.shape(self.Features)}')

        method = train_conf.get('method', 'ETR')

        SearchCV = train_conf.get('SearchCV',None)

        self.selection = False
    
        search = False

        if method == 'ETR':

            regr_model = ExtraTreesRegressor(random_state=self.rnd_seed)

            regr_model.set_params(**params)

            self.normalization = False

            self.selection = False



        if method == 'RFR':

            regr_model = RandomForestRegressor(random_state=self.rnd_seed)

            regr_model.set_params(**params)

            self.normalization = False



        if method == 'SVR':

            single_regr_model = SVR()

            single_regr_model.set_params(**params)

            regr_model = MultiOutputRegressor(single_regr_model)

            self.normalization = True



        if method == 'MLPR':

            regr_model = MLPRegressor()

            regr_model.set_params(**params)

            self.normalization = True


        if SearchCV == 'Random':

            search = True

            regr_model = RandomizedSearchCV(regr_model,param_distributions=params,n_iter=self.n_iter_search,random_state=self.rnd_seed)



        if SearchCV == 'Grid':

            search = True

            regr_model = GridSearchCV(regr_model,param_grid=params)



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


        if self.normalization == True:
            transformer = StandardScaler().fit(self.Features)
            self.Features = transformer.transform(self.Features)

            with open(f'{self.model_name}_transformer.joblib','w') as g:
                g.truncate(0)
            g.close()

            pathname = f'{self.model_name}_transformer.joblib'

            dump(transformer,pathname)
            
            print(f'Transformer is saved in {pathname}.\n')


        if method != 'MLPR':
            regr_model.set_params(n_jobs=self.threads)

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