from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from joblib import parallel_backend
from joblib import dump,load
import json as json 
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

from src.ReadWrite import ReadWrite
import os
import numpy as np
import glob as glob

class Training(ReadWrite):
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

        if self.mode =='homo':
            print(f'Importing Features and Targets of homonuclear model ... ', end="")
            self.files = glob.glob('Model_Homo*.json')
            for f in self.files:
                Feature_temp,Targets_temp = self.import_pickle_FT(f)
                self.Features.extend(Feature_temp)
                self.Targets.extend(Targets_temp)
            self.Features = np.array(self.Features)
            self.Targets = np.array(self.Targets)

            print('done\n')

        elif self.mode == 'hetero':
            print(f'Importing Features and Targets of heteronuclear model... ', end="")

            self.files = glob.glob('Model_Hetero*.json')
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
        if self.mode == 'homo':
            N_est = self.ml_parameter[0].get('n_estimators')
            max_depth = self.ml_parameter[0].get('max_depth')

        if self.mode == 'hetero':
            N_est = self.ml_parameter[1][0].get('n_estimators')
            max_depth = self.ml_parameter[1][0].get('max_depth')            

        print(f'Training {self.mode}nuclear model with a feature matrix of shape {np.shape(self.Features)}... ',end="")

        method = train_conf.get('method', 'ETR')
        SearchCV = train_conf.get('SearchCV',None)

        if method == 'ETR':
            regr_model = ExtraTreesRegressor(max_depth=max_depth,bootstrap=True,random_state=self.rnd_seed,max_features=1.0)
            self.normalization = False
        
        if method == 'RFR':
            regr_model = RandomForestRegressor(n_estimators=N_est,max_depth=max_depth,bootstrap=True,random_state=self.rnd_seed,max_features=1.0)
            self.normalization = False

        if method == 'SVR':
            single_regr_model = SVR(epsilon=1e-3,C=1.0,tol=1e-3,kernel='rbf')
            regr_model = MultiOutputRegressor(single_regr_model)
            self.normalization = True

        if method == 'MLPR':
            regr_model = MLPRegressor(hidden_layer_sizes=300,alpha=0.0001,learning_rate_init=0.0001)
            self.normalization = True


        if SearchCV == 'Random':

            params_grid_path = 'SearchCVParams.json'
            with open(params_grid_path) as f:
                param_grid = json.load(f)
            f.close()

            regr_model = RandomizedSearchCV(regr_model,param_distributions=param_grid.get(method),n_iter=25,random_state=self.rnd_seed)

        if SearchCV == 'Grid':
            params_grid_path = 'SearchCVParams.json'
            with open(params_grid_path) as f:
                param_grid = json.load(f)
            f.close()
            regr_model = GridSearchCV(regr_model,param_grid=param_grid.get(method))

        
        self.Targets = self.Targets.astype(np.float32)
        self.Features = self.Features.astype(np.float32)


        if self.normalization == True:
            transformer = StandardScaler().fit(self.Features)
            self.Features = transformer.transform(self.Features)

            with open(f'{self.model_name}_{self.mode}_transformer.joblib','w') as g:
                g.truncate(0)
            g.close()

            pathname = f'{self.model_name}_{self.mode}_transformer.joblib'
            print(f'done')
            dump(transformer,pathname)
            print(f'Transformer is saved in {pathname}.\n')


        with parallel_backend('threading',n_jobs=self.threads):
            regr_model.fit(self.Features,self.Targets)


        with open(f'{self.model_name}_{self.mode}.joblib','w') as g:
            g.truncate(0)
        g.close()

        pathname = f'{self.model_name}_{self.mode}.joblib'
        print(f'done')
        dump(regr_model,pathname)
        print(f'Model is saved in {pathname}.\n')




        del g
        del regr_model
        #del clf
        del self.Features
        del self.Targets

        if self.normalization:
            del transformer

        return