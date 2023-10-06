import numpy as np
import pandas as pd
from hess_ml.src.Rotation_func import Rotation_Functions
import numpy as np
import pickle as pickle
import os
import json as json
import faulthandler
from ase.io import read as ase_read


class Input:
    def __init__(self):
        pass

    def import_coord(self, file):
        with open(file) as myfile:
            head = [next(myfile) for _ in range(2)]

        coord_var = pd.read_csv(
            file,
            sep="\s+",
            skiprows=2,
            header=None,
            keep_default_na=False,
            na_values=["_"],
        )

        coord_var.columns = ["atoms", "x", "y", "z"]

        return coord_var, head

    def import_dipm(self, file):
        coord_var = pd.read_csv(file, sep=",")

        return coord_var

    def import_hessian(self, file, nat):
        LineList = []

        with open(file, "r") as fd:
            Lines = [line.rstrip("\n") for line in fd]
            for line in Lines[1:]:
                LineList += line.split()

        hess = np.zeros([nat * 3, nat * 3])

        i = 0

        for k in range(len(hess[0, :])):
            for l in range(len(hess[:, 0])):
                hess[k, l] = float(LineList[i])
                i += 1

        return hess

    def import_hessian_dftd4(self, file, coord):
        nat3 = len(coord["atoms"]) * 3

        file_path = os.path.join(self.geo_working_dir, "dftd4.json")

        with open(file_path) as f:
            egh = json.load(f)

        hess_dftd4 = np.array(egh.get("hessian")).reshape(nat3, nat3)

        return hess_dftd4

    def import_gradient(self, file):
        with open(file, "rb") as f:
            f.close()

        self.gradient = np.genfromtxt(
            file, skip_header=2 + self.N_atoms, skip_footer=1, loose=True
        )
        # gradient = gradient.flatten()

        return

    def filter_feature(self):
        self.dipm = {}
        self.qm = {}
        self.q = {}
        self.cn = {}
        self.p = {}

        for orb in ["s", "p", "d", "A", "e", "Z"]:
            if not (orb in {"s", "p", "d"}):
                self.dipm[f"delta_{orb}"] = self.ml_feat.filter(
                    regex=f"delta_dipm_{orb}_"
                ).to_numpy()
                self.qm[f"delta_{orb}"] = self.ml_feat.filter(
                    regex=f"delta_qm_{orb}_"
                ).to_numpy()

            if not (orb in {"e", "Z"}):
                self.dipm[f"{orb}"] = self.ml_feat.filter(
                    regex=f"^dipm_{orb}_"
                ).to_numpy()
                self.qm[f"{orb}"] = self.ml_feat.filter(regex=f"^qm_{orb}_").to_numpy()

                if orb != "A":
                    self.q[f"{orb}"] = self.ml_feat.filter(regex=f"q_{orb}").to_numpy()

        self.energy_based = self.ml_feat.loc[
            :,
            [
                "response",
                "gap",
                "chem.pot",
                "HOAO_a",
                "LUAO_a",
                "HOAO_b",
                "LUAO_b",
                "delta_gap",
                "delta_chem_pot",
                "delta_HOAO",
                "delta_LUAO",
                "E_repulsion",
                "E_EHT",
                "E_disp_2",
                "E_disp_3",
                "E_ies_ixc",
                "E_aes",
                "E_axc",
            ],
        ].to_numpy()

        self.cn["default"] = self.ml_feat.loc[:, "CN"].to_numpy()
        self.cn["delta"] = self.ml_feat.loc[:, "delta_CN"].to_numpy()

        self.p["default"] = self.ml_feat.loc[:, "p_A"].to_numpy()
        self.p["delta"] = self.ml_feat.loc[:, "delta_p_A"].to_numpy()

        return

    def import_ml_features(self):
        if self.config.get("feature", False) == "tblite":
            try:
                from tblite.ase import TBLite
            except:
                print("TBLite is not available.")

            try:
                faulthandler.enable()
                # Use ase to read in coordinates

                mol = ase_read(filename=os.path.join(self.folder, self.xyz_file))
                # create a ase calculator instance
                # xtbml value is used to compute the features
                # 0 = compute no xtbml features
                # 1 = all multipoles are given as norms
                # 2 = multipoles are exported as vectors

                # final single point with xtbml features
                charge = 0
                uhf = 0
                ChargePath = os.path.join(self.folder, ".CHRG")
                if os.path.isfile(ChargePath):
                    with open(ChargePath, "r") as chrg:
                        charge = int(chrg.readline())

                uhfFilePath = os.path.join(self.folder, ".UHF")

                if os.path.isfile(uhfFilePath):
                    with open(uhfFilePath, "r") as uhfFile:
                        uhf = int(uhfFile.readline())

                mol.calc = TBLite(method="GFN2-xTB", xtbml=2, charge=charge, uhf=uhf)

                mol.calc.calculate(mol)

                # from the results type various properties can be retrived
                # "xtbml" = xtbml fetaures as a numpy array (natoms, nfeatures)
                # "xtbml weights" = get Mulliken-based partitioning weights
                # "xtbml labels" = get the labels corresponding to the features
                X = mol.calc.results["xtbml"]
                w = mol.calc.results["xtbml weights"]
                labels = mol.calc.results["xtbml labels"]

                self.ml_feat = pd.DataFrame(X, columns=labels)
                self.ml_feat["weights"] = w

            except:
                print("SCF did not converge, no prediction will be done.")
                self.do_calc = False

        else:
            self.ml_feat = pd.read_csv(os.path.join(self.folder, self.file_feature))

        return

    def import_ml_features_legacy(self, file):
        GFN2_quantities = pd.read_csv(f"{file}")

        self.CN = np.array(
            GFN2_quantities.loc[
                :, ["coordination number", "delta coordination number"]
            ].values.tolist()
        )

        self.q_atom = np.array(
            GFN2_quantities.loc[
                :, ["atomic partial charges", "delta partial charges"]
            ].values.tolist()
        )

        self.dipm_atom = np.array(
            GFN2_quantities.loc[
                :, ["dipm_atom_x", "dipm_atom_y", "dipm_atom_z"]
            ].values.tolist()
        )
        self.dipm_delta = np.array(
            GFN2_quantities.loc[
                :, ["dipm_delta_x", "dipm_delta_y", "dipm_delta_z"]
            ].values.tolist()
        )

        self.dipm_only_mull = np.array(
            GFN2_quantities.loc[
                :,
                [
                    "delta dipm only mull x",
                    "delta dipm only mull y",
                    "delta dipm only mull z",
                ],
            ].values.tolist()
        )
        self.dipm_only_Z = np.array(
            GFN2_quantities.loc[
                :, ["delta dipm only Z x", "delta dipm only Z y", "delta dipm only Z z"]
            ].values.tolist()
        )

        self.qm_atom = Rotation_Functions.qm_matrix(
            np.array(
                GFN2_quantities.loc[
                    :,
                    [
                        "qm_atom_xx",
                        "qm_atom_yy",
                        "qm_atom_zz",
                        "qm_atom_xy",
                        "qm_atom_xz",
                        "qm_atom_yz",
                    ],
                ].values.tolist()
            )
        )
        self.qm_delta = Rotation_Functions.qm_matrix(
            np.array(
                GFN2_quantities.loc[
                    :,
                    [
                        "qm_delta_xx",
                        "qm_delta_yy",
                        "qm_delta_zz",
                        "qm_delta_xy",
                        "qm_delta_xz",
                        "qm_delta_yz",
                    ],
                ].values.tolist()
            )
        )

        self.qm_delta_only_mull = np.array(
            GFN2_quantities.loc[
                :,
                [
                    "delta qm only mull x",
                    "delta qm only mull y",
                    "delta qm only mull z",
                ],
            ].values.tolist()
        )
        self.qm_delta_only_Z = np.array(
            GFN2_quantities.loc[
                :, ["delta qm only Z x", "delta qm only Z y", "delta qm only Z z"]
            ].values.tolist()
        )

        self.energy_based = np.array(
            GFN2_quantities.loc[
                :,
                [
                    "chem pot",
                    "HOAO_a (eV)",
                    "LUAO_a (eV)",
                    "HOAO_b (eV)",
                    "LUAO_b (eV)",
                    "E_repulsion",
                    "E_EHT",
                    " E_disp_2",
                    "E_disp_3",
                    "E_ies_ixc",
                    "E_aes",
                    "E_tot",
                    "E_axc",
                    " chem_pot_ext",
                    "e_gap_ext",
                    "ehoao_ext",
                    "eluao_ext",
                ],
            ].values.tolist()
        )

        self.names = GFN2_quantities.columns.tolist()

        return

    def import_wbo(self, file):
        wbo = pd.read_csv(file, names=["at1", "at2", "wbo"], sep="\s+")
        return wbo

    def import_pickle_FT_old(self, file):
        feature = []
        target = []

        i = 0

        with open(f"{file}", "rb") as f:
            while True:
                try:
                    i += 1
                    temp_obj = pickle.load(f)
                    feature.extend(temp_obj["Feature"])
                    target.extend(temp_obj["Target_AB"])
                except EOFError:
                    # print(f'Features and Targets of a total of {i-1} structures are used.\n')
                    break

        return feature, target

    def import_pickle_FT(self, file):
        i = 0
        with open(f"{file}", "rb") as f:
            i += 1
            temp_obj = pickle.load(f)
            feature = temp_obj["Feature"]
            target = temp_obj["Target_AB"]

        return feature, target

    def rd_txt_file(self, file):
        with open(f"{file}", "rb") as f:
            filenames = f.read().splitlines()
        f.close()

        for i in range(len(filenames)):
            filenames[i] = filenames[i].decode("ascii")

        return filenames

    def truncate_file(self, file):
        if os.path.isfile(file):
            with open(file, "wb+") as f:
                f.truncate(0)
            f.close()
        return


class Output:
    def __init__(self):
        pass

    def hessian_to_xtb(self, file, hessian):
        """
        Writes a hessian from a numpy array to a xtb format hessian
        Requires file name and the array
        """

        Nat3 = len(hessian)

        with open(file, "w+") as myfile:
            myfile.write(f"$hessian\n")

            for i in range(Nat3):
                str_list = [f"{x: 10.10f}" for x in hessian[i]]

                for k in range(0, Nat3, 5):
                    sep = "\t"

                    sep = sep.join(str_list[k : k + 5])
                    myfile.write("\t")
                    myfile.write(sep)
                    myfile.write("\n")
        myfile.close()

        return

    def data_to_txt(self, data, file):
        with open(file, "w+") as f:
            for d in data:
                f.write(d)
                f.write("\n")
        f.close()

        return
