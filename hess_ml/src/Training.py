
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from joblib import dump

import json as json 

from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from hess_ml.src.IO import Input
import os
import numpy as np
import glob as glob

class Training(Input):
    def __init__(self) -> None:
        super().__init__
        pass

    def train(self,train_conf,mode=None):
        self.mode = mode
        self.import_FT()
        self.training_model(train_conf)
        
        # if mode == 'hetero':
        #     final_file = 'Model_Hetero.json'
        # if mode == 'homo':
        #     final_file = 'Model_Homo.json'
        
        # self.merge_JsonFiles(self.files,final_file)

        return

    def import_FT(self):
        
        self.Features = []
        self.Targets = []

        if self.mode == 'hetero':
            print(f'Importing Features and Targets of heteronuclear model... ', end="")

            self.files = glob.glob('Model_*.json')
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

    def training_model(self,train_conf):

        params = self.config['training_testing'][self.mode]
        
        if type(params) == list:
            params = params[0]

        print(params)
        print(f'Training {self.mode}nuclear model with a feature matrix of shape {np.shape(self.Features)}')

        method = train_conf.get('method', 'ETR')
        SearchCV = train_conf.get('SearchCV',None)

        self.selection = False


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

        search = False

        if SearchCV == 'Random':
            search = True
            regr_model = RandomizedSearchCV(regr_model,param_distributions=params,n_iter=self.n_iter_search,random_state=self.rnd_seed)

        if SearchCV == 'Grid':
            search = True
            regr_model = GridSearchCV(regr_model,param_grid=params)

        
        self.Targets = self.Targets.astype(np.float32)
        self.Features = self.Features.astype(np.float32)
        
        print(f'Feature vector shape {self.Features.shape}')

        if self.selection:
            selector = VarianceThreshold(threshold=0.1*(1-0.1))
            self.Features = selector.fit_transform(self.Features)

            with open(f'{self.model_name}_{self.mode}_selector.joblib','w') as g:
                g.truncate(0)
            g.close()

            pathname = f'{self.model_name}_{self.mode}_selector.joblib'

            dump(selector,pathname)
            print(f'Selector is saved in {pathname}.\n')

            print(f'Feature vector reduced to shape {self.Features.shape}')


        if self.normalization == True:
            transformer = StandardScaler().fit(self.Features)
            self.Features = transformer.transform(self.Features)

            with open(f'{self.model_name}_{self.mode}_transformer.joblib','w') as g:
                g.truncate(0)
            g.close()

            pathname = f'{self.model_name}_{self.mode}_transformer.joblib'

            dump(transformer,pathname)
            print(f'Transformer is saved in {pathname}.\n')




        if method != 'MLPR':
            regr_model.set_params(n_jobs=self.threads)

        regr_model.fit(self.Features,self.Targets)

        
        if search:
            print(f'Used Parameters:\n {regr_model.best_params_}')

        else:
            print(f'Used Parameters:\n {regr_model.get_params()}')

        with open(f'{self.model_name}_{self.mode}.joblib','w') as g:
            g.truncate(0)
        g.close()

        pathname = f'{self.model_name}_{self.mode}.joblib'
        dump(regr_model,pathname)
        print(f'Model is saved in {pathname}.\n')




        del g
        del regr_model
        #del clf
        del self.Features
        del self.Targets

        if self.normalization:
            del transformer

        if self.selection:
            del selector
        return