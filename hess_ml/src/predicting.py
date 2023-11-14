import copy
import glob
import os
import pickle
import time

import numpy as np
from joblib import load

from hess_ml.src.io import Input, Output
from hess_ml.src.observables import Observables
from hess_ml.src.template import TestMLHessianGFN2xTB, TestMLHessianORCA


class Predicting(Input, Output, Observables):
    def __init__(self) -> None:
        super().__init__()

    def predict(self, files, folder="",model=None):

        if model is None:
            self.model = load(os.path.join(folder, f"{self.model_name}.joblib"))

        else:
            self.model = model

        self.not_considered = []

        for program in self.config["general"].get("program", ["xTB"]):
            if len(self.config["general"].get("program", ["xTB"])) == 1:
                if program.lower()  == "orca":
                    self.molgen = TestMLHessianORCA()
                elif program.lower()  == "xtb":
                    self.molgen = TestMLHessianGFN2xTB()
            else:
                print("More than one program not implemented yet!")

        self.molgen = TestMLHessianGFN2xTB()

        for file in range(len(files)):
            self.predict_hessian(folder=files[file])

        with open("not_considered_pred", "w") as outfile:
            outfile.write("\n".join(str(i) for i in self.not_considered))
        outfile.close()

    def error_estimation(self, folders, rnd_seed, train_size):
        print("Computing error on test set")

        size = 0
        error = 0

        for program in self.config["general"].get("program", ["xTB"]):
            if len(self.config["general"].get("program", ["xTB"])) == 1:
                if program.lower()  == "orca":
                    self.molgen = TestMLHessianORCA()
                elif program.lower()  == "xtb":
                    self.molgen = TestMLHessianGFN2xTB()
            else:
                print("More than one program not implemented yet!")

        for folder in folders:
            if folder not in self.not_considered:
                mol = copy.deepcopy(self.molgen)
                mol.setConfiguration(folder, self.config["molecule"])
                mol.hessians_difference(self.config["molecule"]["target_file"], "MLhessian")
                shape = np.shape(mol.hess_diff)
                size += shape[0] * shape[1]
                error += np.sum(mol.hess_diff**2)

        error = np.sqrt(error / size)

        print("Seed\tTrain Size\tRMSD")
        print(f"{rnd_seed}\t{train_size*100: 3.0f}\t{error : 0.5f}")

        with open("results","a+") as file:
            file.write(f"{rnd_seed}\t{train_size*100: 3.0f}\t{error : 0.5f}\n")

    def predict_hessian(self, folder):
        mol = copy.deepcopy(self.molgen)
        mol.setConfiguration(folder, self.config["molecule"])
        mol.ProcessData(model=self.model)
