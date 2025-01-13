from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING

import numpy as np
from joblib import load

from hess_ml.src.io import Input, Output
from hess_ml.src.observables import Observables
from hess_ml.src.Targets.hessian import xTBHessTarget
from hess_ml.src.template import TestMLHessianGFN2xTB, TestMLHessianORCA

if TYPE_CHECKING:
    from hess_ml.src.environment import Environment

class Predicting(Input, Output, Observables):
    def __init__(self) -> None:
        super().__init__()

    def predict(self:Environment, files, folder="",model=None):

        if model is None:
            self.model = load(os.path.join(folder, f"{self.model_name}.joblib"))

        else:
            self.model = model

        self.not_considered = []

        for program in self.config.molecule.program:
            if len(self.config.molecule.program) == 1:
                if program.lower()  == "orca":
                    self.molgen = TestMLHessianORCA()
                elif program.lower()  == "xtb":
                    self.molgen = TestMLHessianGFN2xTB()
            else:
                print("More than one program not implemented yet!")

        size = 0
        error = 0

        for file in range(len(files)):
            self.predict_hessian(folder=files[file])

            if self.config.interal.train:
                target_file = os.path.join(files[file], self.config.molecule.target_file)
                read = xTBHessTarget(target_file, self.mol.N_atoms)
                read.ImportTarget()

                hess_diff = self.mol.predHess - read.target

                shape = np.shape(hess_diff)
                size += shape[0]**2
                error += np.sum(hess_diff**2)

        if self.config.interal.train:

            error = np.sqrt(error / size)

            print("Seed\tTrain Size\tRMSD")
            print(f"{self.rnd_seed}\t{self.train_size*100: 3.3f}\t{error : 0.5f}")

            with open("results","a+") as file:
                file.write(f"{self.rnd_seed}\t{self.train_size*100: 3.3f}\t{error : 0.5f}\n")

        with open("not_considered_pred", "w") as outfile:
            outfile.write("\n".join(str(i) for i in self.not_considered))
        outfile.close()

        del self.model

    def error_estimation(self:Environment, folders, rnd_seed, train_size):
        print("Computing error on test set")

        size = 0
        error = 0

        for program in self.config.molecule.program:
            if len(self.config.molecule.program) == 1:
                if program.lower()  == "orca":
                    self.molgen = TestMLHessianORCA()
                elif program.lower()  == "xtb":
                    self.molgen = TestMLHessianGFN2xTB()
            else:
                print("More than one program not implemented yet!")

        for folder in folders:
            if folder not in self.not_considered:
                mol = copy.deepcopy(self.molgen)
                mol.setConfiguration(folder, self.config.general,self.config.molecule)
                mol.hessians_difference(self.config.molecule.target_file, "MLhessian")
                shape = np.shape(mol.hess_diff)
                size += shape[0]**2
                error += np.sum(mol.hess_diff**2)

        error = np.sqrt(error / size)

        print("Seed\tTrain Size\tRMSD")
        print(f"{rnd_seed}\t{train_size*100: 3.0f}\t{error : 0.5f}")

        with open("results","a+") as file:
            file.write(f"{rnd_seed}\t{train_size*100: 3.0f}\t{error : 0.5f}\n")

    def predict_hessian(self:Environment, folder):
        self.mol = copy.deepcopy(self.molgen)
        self.mol.setConfiguration(folder,self.config.general,self.config.molecule)
        self.mol.ProcessData(model=self.model)


    def optimization(self:Environment,folder):
        self.model = load(os.path.join("", "2MDhess.joblib"))
        self.mol = TestMLHessianGFN2xTB()

        self.mol.setConfiguration(folder,self.config.general,self.config.molecule)

        self.mol.ProcessData(model=self.model)

        self.mol.optimize_step()

