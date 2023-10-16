import os

import numpy as np
import pandas as pd
from ase.io import read as ase_read


class FeatureTBlite:
    def __init__(self) -> None:
        return

    def ImportFeature(self):
        # Use ase to read in coordinates
        from tblite.ase import TBLite

        try:
            ase_mol = ase_read(filename=self.xyz_file)
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
                with open(ChargePath) as chrg:
                    charge = int(chrg.readline())

            uhfFilePath = os.path.join(self.folder, ".UHF")

            if os.path.isfile(uhfFilePath):
                with open(uhfFilePath) as uhfFile:
                    uhf = int(uhfFile.readline())

            ase_mol.calc = TBLite(method="GFN2-xTB", xtbml=2, charge=charge, uhf=uhf)

            ase_mol.calc.calculate(ase_mol)

            # from the results type various properties can be retrived
            # "xtbml" = xtbml fetaures as a numpy array (natoms, nfeatures)
            # "xtbml weights" = get Mulliken-based partitioning weights
            # "xtbml labels" = get the labels corresponding to the features
            X = ase_mol.calc.results["xtbml"]
            w = ase_mol.calc.results["xtbml weights"]
            labels = ase_mol.calc.results["xtbml labels"]

            self.ml_feat = pd.DataFrame(X, columns=labels)
            self.ml_feat["weights"] = w

            self.ReadGradient(self.gradient_file)
            self.FilterFeatures()

        except:
            self.do_calc = False

            print("No convergenve structure will not be considered.")

    def ReadGradient(self, file):
        with open(file, "rb") as f:
            f.close()

        self.gradient = np.genfromtxt(
            file,
            skip_header=2 + self.N_atoms,
            skip_footer=1,
            loose=True,
        )

    def FilterFeatures(self):
        self.dipm = {}
        self.qm = {}
        self.q = {}
        self.cn = {}
        self.p = {}

        for orb in ["s", "p", "d", "A", "e", "Z"]:
            if orb not in {"s", "p", "d"}:
                self.dipm[f"delta_{orb}"] = self.ml_feat.filter(
                    regex=f"delta_dipm_{orb}_",
                ).to_numpy()
                self.qm[f"delta_{orb}"] = self.ml_feat.filter(
                    regex=f"delta_qm_{orb}_",
                ).to_numpy()

            if orb not in {"e", "Z"}:
                self.dipm[f"{orb}"] = self.ml_feat.filter(
                    regex=f"^dipm_{orb}_",
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
