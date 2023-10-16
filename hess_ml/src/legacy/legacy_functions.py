import os
import pickle

import numpy as np
from joblib import load


def parse_folders(self, folder, subfolder):
    print(f"Gathering Folders from {folder}.")

    if subfolder:
        self.gather_subfolders(folder)

    else:
        self.gather_folders(folder)


def gather_subfolders(self, folder):
    self.total_structures = 0

    self.geo_dir = []

    molecule_dir = sorted(
        [
            mol
            for mol in os.listdir(f"{folder}")
            if os.path.isdir(os.path.join(f"{folder}", mol))
        ],
    )

    for mol in range(len(molecule_dir)):
        data_dir = os.path.join(f"{folder}", molecule_dir[mol])

        temp_dir = sorted(
            [
                os.path.join(data_dir, geo)
                for geo in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, geo))
            ],
        )

        self.geo_dir.extend(temp_dir)

        self.total_structures += len(temp_dir)


def gather_folders(self, folder):
    molecule_dir = sorted(
        [
            mol
            for mol in os.listdir(f"{folder}")
            if os.path.isdir(os.path.join(f"{folder}", mol))
        ],
    )

    self.geo_dir = molecule_dir

    self.total_structures = len(self.geo_dir)


def predict_hess_depracted(self):
    # _____reads heteronuclear model and predicts for each structure the heteronuclear blocks____

    self.truncate_file("PredData.json")

    het_model = load(f"{self.model_name}.joblib")

    if self.normalization:
        pathname = f"{self.model_name}_transformer.joblib"
        transformer = load(pathname)

    if self.selection:
        pathname = f"{self.model_name}_selector.joblib"
        selector = load(pathname)

    test_files = glob.glob("TestData_*.json")
    for file in test_files:
        with open(f"{file}", "rb") as f:
            while True:
                try:
                    temp_obj = pickle.load(f)

                    if self.normalization:
                        H_hetero = het_model.predict(
                            transformer.transform(np.array(temp_obj.get("Feature"))),
                        )

                    elif self.selection:
                        H_hetero = het_model.predict(
                            selector.transform(np.array(temp_obj.get("Feature"))),
                        )

                    else:
                        H_hetero = het_model.predict(
                            np.array(temp_obj.get("Feature")),
                        )

                    temp_obj["pred_target_AB"] = H_hetero

                    with open("PredData.json", "ab") as g:
                        pickle.dump(temp_obj, g)

                except EOFError:
                    break
        g.close()
        f.close()

    del het_model


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
