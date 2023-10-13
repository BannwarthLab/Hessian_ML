from joblib import load
import pickle as pickle

import numpy as np
import glob as glob
import os

from hess_ml.src.IO import Input
from hess_ml.src.IO import Output
from hess_ml.src.Template import TestMLHessianGFN2xTB
from hess_ml.src.Observables import Observables
import time as time
import copy

class Predicting(Input, Output,Observables):
    def __init__(self) -> None:
        super().__init__
        pass

    def predict(self, files, folder=""):

        #try:
        #    self.predict_model
        #    self.model = load(os.path.join(folder, f"{self.predict_model}.joblib"))
        #except:
        self.model = load(os.path.join(folder, f"{self.model_name}.joblib"))
            

        if self.config.get("predict", {"normalization": False}).get(
            "normalization"
        ):  # self.normalization
            pathname = os.path.join(folder, f"{self.model_name}_transformer.joblib")

            self.transformer = load(pathname)

            pathname = os.path.join(
                folder, f"{self.model_name}_transformer_target.joblib"
            )

            self.target_transformer = load(pathname)

        if self.config.get("predict", {"selection": False}).get("selection"):
            pathname = os.path.join(folder, f"{self.model_name}_selector.joblib")

            self.selector = load(pathname)

        self.not_considered = []

        self.molgen = TestMLHessianGFN2xTB()
        
        for file in range(len(files)):
            self.predict_hessian(folder=files[file])

        with open("not_considered_pred", "w") as outfile:
            outfile.write("\n".join(str(i) for i in self.not_considered))
        outfile.close

        return

    def error_estimation(self, folders, rnd_seed, train_size):
        print("Computing error on test set")

        size = 0
        error = 0
        self.molgen = TestMLHessianGFN2xTB()
    
        for folder in folders:
            mol = copy.deepcopy(self.molgen) 
            mol.setConfiguration(folder, self.config["molecule"])
            mol.hessians_difference(self.config["molecule"]["target_file"], "MLhessian")
            shape = np.shape(mol.hess_diff)
            size += shape[0] * shape[1]
            error += np.sum(mol.hess_diff**2)

        error = np.sqrt(error / size)

        print("Seed\tTrain Size\tRMSD")
        print(f"{rnd_seed}\t{train_size*100: 3.0f}\t{error : 0.5f}")

        return
    
    def predict_hessian(self, folder):

        
        mol = copy.deepcopy(self.molgen) 
        mol.setConfiguration(folder, self.config["molecule"])
        mol.ProcessData(model=self.model)
 
        return


