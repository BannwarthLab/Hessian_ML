from sklearn.ensemble import ExtraTreesRegressor
from joblib import parallel_backend
from joblib import dump,load

from ReadWrite import ReadWrite
import os

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
            self.Feature= self.import_pickle_FT('Feature_Vector_Homo')
            self.Target = self.import_pickle_FT('Target_Vector_Homo')

        elif self.mode == 'hetero':
            self.Feature = self.import_pickle_FT('Feature_Vector_Hetero')
            self.Target = self.import_pickle_FT('Target_Vector_Hetero')
        else:
            print('no feature or target imported')

        return 

    def training_model(self):
        if self.mode == 'homo':
            N_est = self.ml_parameter[0].get('n_estimators')
            max_depth = self.ml_parameter[0].get('max_depth')

        if self.mode == 'hetero':
            N_est = self.ml_parameter[1].get('n_estimators')
            max_depth = self.ml_parameter[1].get('max_depth')

        regr_model = ExtraTreesRegressor(n_estimators=N_est,max_depth=max_depth,bootstrap=True,random_state=self.rnd_seed)
        with parallel_backend('threading',n_jobs=self.threads):
            regr_model.fit(self.Feature,self.Target)

        pathname = f'{self.model_name}{self.mode}.joblib'

        dump(regr_model,pathname)

        del regr_model
        del self.Feature
        del self.Target

        return