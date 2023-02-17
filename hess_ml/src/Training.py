from sklearn.ensemble import ExtraTreesRegressor
from joblib import parallel_backend
from joblib import dump,load

from src.ReadWrite import ReadWrite
import os
import numpy as np
import glob as glob

class Training(ReadWrite):
    def __init__(self) -> None:
        super().__init__
        pass

    def train(self,mode=None):
        self.mode = mode
        self.import_FT()
        self.training_model()
        
        # if mode == 'hetero':
        #     final_file = 'Model_Hetero.json'
        # if mode == 'homo':
        #     final_file = 'Model_Homo.json'
        
        # self.merge_JsonFiles(self.files,final_file)


        return

    def import_FT(self):

        if self.mode =='homo':
            print(f'Importing Features and Targets of homonuclear model ... ', end="")
            self.files = glob.glob('Model_Homo*.json')
            for f in self.files:
                self.Features,self.Targets = self.import_pickle_FT(f)
            print('done\n')

        elif self.mode == 'hetero':
            print(f'Importing Features and Targets of heteronuclear model... ', end="")

            self.files = glob.glob('Model_Hetero*.json')
            for f in self.files:
                self.Features,self.Targets = self.import_pickle_FT(f)
            print('done\n')

        else:
            print('no feature or target imported')


        return 

    def training_model(self):
        if self.mode == 'homo':
            N_est = self.ml_parameter[0].get('n_estimators')
            max_depth = self.ml_parameter[0].get('max_depth')

        if self.mode == 'hetero':
            N_est = self.ml_parameter[1][0].get('n_estimators')
            max_depth = self.ml_parameter[1][0].get('max_depth')

        print(f'Training {self.mode}nuclear model... ',end="")
        regr_model = ExtraTreesRegressor(n_estimators=N_est,max_depth=max_depth,bootstrap=True,random_state=self.rnd_seed)

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
        del self.Features
        del self.Targets

        return