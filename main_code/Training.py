from sklearn.ensemble import ExtraTreesRegressor
from joblib import parallel_backend
from joblib import dump,load

from ReadWrite import ReadWrite
import os
import numpy as np

class Training(ReadWrite):
    def __init__(self) -> None:
        super().__init__
        pass

    def train(self,mode=None):
        self.mode = mode
        self.import_FT()
        self.training_model()
        return

    def import_FT(self):

        if self.mode =='homo':
            self.Features,self.Targets = self.import_pickle_FT2('Model_Homo.json')

        elif self.mode == 'hetero':
            self.Features,self.Targets = self.import_pickle_FT2('Model_Hetero.json')

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

        regr_model = ExtraTreesRegressor(n_estimators=N_est,max_depth=max_depth,bootstrap=True,random_state=self.rnd_seed)

        with parallel_backend('threading',n_jobs=self.threads):
            regr_model.fit(self.Features,self.Targets)

        with open(f'{self.model_name}_{self.mode}.joblib','w') as g:
            g.truncate(0)
            g.close()

        pathname = f'{self.model_name}_{self.mode}.joblib'

        dump(regr_model,pathname)

        del g
        del regr_model
        del self.Features
        del self.Targets

        return