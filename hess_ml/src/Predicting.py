from joblib import load
import pickle as pickle

import numpy as np
import glob as glob
import os

from hess_ml.src.Observables import Observables
from hess_ml.src.IO import Input
from hess_ml.src.IO import Output
from hess_ml.src.Geometry import Geometry
from joblib import Parallel, delayed
import time as time


class Predicting(Input, Output, Observables):
    def __init__(self) -> None:
        super().__init__
        pass

    def predict(self, files, folder=""):
        try:
            self.predict_model
            self.model = load(os.path.join(folder, f"{self.predict_model}.joblib"))
        except:
            self.model = load(os.path.join(folder, f"{self.model_name}.joblib"))
            pass

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

        Parallel(n_jobs=1)(
            delayed(self.predict_hessian)(file=files[file])
            for file in range(len(files))
        )

        with open("not_considered_pred", "w") as outfile:
            outfile.write("\n".join(str(i) for i in self.not_considered))
        outfile.close

        return

    def error_estimation(self, folders, rnd_seed, train_size):
        print("Computing error on test set")

        size = 0
        error = 0
        for folder in folders:
            mol = Geometry(folder, self.config["geometry"])
            mol.hessians_difference(self.config["geometry"]["target_file"], "MLhessian")
            shape = np.shape(mol.hess_diff)
            size += shape[0] * shape[1]
            error += np.sum(mol.hess_diff**2)

        error = np.sqrt(error / size)

        print("Seed\tTrain Size\tRMSD")
        print(f"{rnd_seed}\t{train_size*100: 3.0f}\t{error : 0.5f}")

        return

    def comp_test_observables(self):
        freq_pred_list = []
        ZPE_pred = []
        Z_pred = []

        freq_true_list = []
        ZPE_true = []
        Z_true = []

        with open("pred_structures_final.json", "rb") as f:
            i = 0

            while True:
                i += 1
                try:
                    temp_obj = pickle.load(f)
                    self.R_MI_APF_mat = temp_obj.get("R_MI_APF_mat")
                    self.xyz = temp_obj.get("xyz")
                    self.transpose_list = temp_obj.get("transpose_list")

                    self.N_atoms = temp_obj.get("N_atoms")

                    hess_vec_ab = np.array(temp_obj.get("pred_target_AB"))

                    freq = self.get_Frequencies(hess_vec_ab)

                    ZPE = self.get_ZPE(freq)

                    Z = self.get_partition_func(freq)

                    # ZPE_harm = self.get_harmonic_ZPE(freq)

                    Z_pred.append(Z)

                    # ZPE_harm_pred.append(ZPE_harm)

                    ZPE_pred.append(ZPE)

                    freq_pred_list.extend(freq)

                    hess_vec_aa = np.array(temp_obj.get("Target_AA"))
                    hess_vec_ab = np.array(temp_obj.get("Target_AB"))

                    freq = self.get_Frequencies(hess_vec_ab, hess_vec_aa)

                    ZPE = self.get_ZPE(freq)
                    Z = self.get_partition_func(freq)

                    # ZPE_harm = self.get_harmonic_ZPE(freq)
                    Z_true.append(Z)

                    # ZPE_harm_true.append(ZPE_harm)
                    ZPE_true.append(ZPE)
                    freq_true_list.extend(freq)

                except EOFError:
                    break

        np.savetxt("pred_frequencies.txt", freq_pred_list)
        np.savetxt("true_frequencies.txt", freq_true_list)

        np.savetxt("pred_ZPEs.txt", ZPE_pred)
        np.savetxt("true_ZPEs.txt", ZPE_true)

        np.savetxt("pred_Z.txt", Z_pred)
        np.savetxt("true_Z.txt", Z_true)

        print("done")

        return

    def predict_hessian(self, file):
        mol = Geometry(file, self.config["molecule"])

        mol.gen_data(1)

        if mol.do_calc:
            cur_time = time.time()

            if self.config["predict"].get("selection", False):  # self.selection
                H_hetero = self.model.predict(
                    self.selector.transform(np.array(mol.Feature_AB))
                )

            if self.config["predict"].get("normalization", False):
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

            self.hessian_to_xtb(os.path.join(file, f"MLhessian"), predHess)

            print(f"Prediction: {time.time()- cur_time: 0.2f} s")

            del H_hetero
            del transpose_list
            del N_atoms
            del R_MI_APF_mat

        return
