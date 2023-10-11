from joblib import load
import pickle as pickle

import numpy as np
import glob as glob
import os

from hess_ml.src.IO import Input
from hess_ml.src.IO import Output
from hess_ml.src.Processing import PredictProcess
from hess_ml.src.Template import TestMLHessianGFN2xTB
from hess_ml.src.Observables import Observables
from joblib import Parallel, delayed
import time as time
from hess_ml.src.decorator.decorator import checkTiming

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
        for folder in folders:
            mol = TestMLHessianGFN2xTB()
            mol.setConfiguration(folder, self.config["molecule"])
            mol.ProcessData()
            mol.hessians_difference(self.config["molecule"]["target_file"], "MLhessian")
            shape = np.shape(mol.hess_diff)
            size += shape[0] * shape[1]
            error += np.sum(mol.hess_diff**2)

        error = np.sqrt(error / size)

        print("Seed\tTrain Size\tRMSD")
        print(f"{rnd_seed}\t{train_size*100: 3.0f}\t{error : 0.5f}")

        return
    
    def gen_hess_from_vec_pred(
        self, hess_vec_ab, N_atoms, R_MI_APF_mat, transpose_list
    ):
        ite_hetero = 0

        Hessian = np.zeros([N_atoms * 3, N_atoms * 3])

        for atom_A in range(N_atoms):
            for atom_B in range(atom_A + 1, N_atoms):
                transpose = False

                if [atom_A, atom_B] in transpose_list:
                    transpose = True

                Hessian = self.fill_matrix_block_AB(
                    hess_vec_ab[ite_hetero],
                    Hessian,
                    R_mat=R_MI_APF_mat,
                    A=atom_A,
                    B=atom_B,
                    transpose=transpose,
                )
                ite_hetero += 1

        for atom_A in range(N_atoms):
            for atom_B in range(N_atoms):
                if atom_A != atom_B:
                    Hessian[
                        3 * atom_A : 3 * atom_A + 3, 3 * atom_A : 3 * atom_A + 3
                    ] -= Hessian[
                        3 * atom_A : 3 * atom_A + 3, 3 * atom_B : 3 * atom_B + 3
                    ]

            Hessian[3 * atom_A : 3 * atom_A + 3, 3 * atom_A : 3 * atom_A + 3] = (
                Hessian[3 * atom_A : 3 * atom_A + 3, 3 * atom_A : 3 * atom_A + 3]
                + np.transpose(
                    Hessian[3 * atom_A : 3 * atom_A + 3, 3 * atom_A : 3 * atom_A + 3]
                )
            ) / 2

        return Hessian
    
    @checkTiming
    def predict_hessian(self, folder):

        mol = TestMLHessianGFN2xTB()
        mol.setConfiguration(folder, self.config["molecule"])
        mol.ProcessData()

        if mol.do_calc:
            cur_time = time.time()

            if self.config.get("predict", {"selection": False}).get("selection", False):  # self.selection
                H_hetero = self.model.predict(
                    self.selector.transform(np.array(mol.Feature_AB))
                )

            if self.config.get("predict", {"selection": False}).get("normalization", False):
                H_hetero = self.model.predict(
                    self.transformer.transform(np.array(mol.Feature_AB))
                )

                H_hetero = self.target_transformer.inverse_transform(H_hetero)

            else:
                H_hetero = self.model.predict((np.array(mol.Feature_AB)))

            transpose_list = mol.transpose_list

            R_MI_APF_mat = mol.R_MI_APF_mat

            N_atoms = mol.N_atoms

            predHess = self.gen_hess_from_vec_pred(
                H_hetero, N_atoms, R_MI_APF_mat, transpose_list
            )

            self.hessian_to_xtb(os.path.join(folder, f"MLhessian"), predHess)

            print(f"Prediction: {time.time()- cur_time: 0.2f} s")

            del H_hetero
            del transpose_list
            del N_atoms
            del R_MI_APF_mat

        return


