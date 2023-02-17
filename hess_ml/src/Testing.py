from sklearn.ensemble import ExtraTreesRegressor
from joblib import dump,load
import pickle as pickle
from src.ReadWrite import ReadWrite
import numpy as np
from src.SaveDat import PickleData
import glob as glob
import os 

class Testing(ReadWrite):
    def __init__(self) -> None:
        super().__init__
        pass

    def test(self):
        self.predict_hess()
        return

    #obj = self.import_pickle_obj(file)

    def predict_hess(self):
        #_____reads heteronuclear model and predicts for each structure the heteronuclear blocks____

        self.truncate_file('pred_structures.json')
        self.truncate_file('pred_structures_final.json')

        with open(f'{self.model_name}_hetero.joblib','rb') as f1:
            het_model = load(f1)
        f1.close()

        test_files = glob.glob('test_structures*.json')
        for file in test_files:
            with open(f'{file}','rb') as f:
                while True:
                    try:
                        temp_obj = pickle.load(f)

                        H_hetero = het_model.predict(np.array(temp_obj.Feature_AB))

                        temp_obj.add_pred_target_AB(H_hetero)
                        with open('pred_structures.json','ab') as g:
                            pickle.dump(temp_obj,g)

                    except EOFError:
                            break
            g.close()
            f.close()

        del het_model

        with open(f'{self.model_name}_homo.joblib','rb') as f:
            hom_model = load(f)

        f.close()

        with open('pred_structures.json','rb') as f:
                while True:
                    try:
                        temp_obj = pickle.load(f)

                        H_hom = hom_model.predict(np.array(temp_obj.Feature_AA))
                        temp_obj.add_pred_target_AA(H_hom)

                        with open('pred_structures_final.json','ab') as g:
                            pickle.dump(temp_obj,g)

                    except EOFError:
                            break
        g.close()
        f.close()
        del hom_model

        for f1 in test_files:
            os.remove(f1)
        os.remove('pred_structures.json')

        return
