import faulthandler
import os

import numpy as np
import pandas as pd
from ase.io import read
from ase.io import read as ase_read
from ase.units import Bohr
from tblite.interface import Calculator

strings = [
        "response",
        "gap",
        "chem_pot",
        "HOAO_a",
        "LUAO_a",
        "HOAO_b",
        "LUAO_b",
        "delta_gap",
        "delta_chem_pot",
        "delta_HOAO",
        "delta_LUAO",
        "E_rep",
        "E_EHT",
        "E_disp2",
        "E_disp3",
        "E_ies_ixc",
        "E_AES",
        "E_AXC",
    ]

pattern = ""

for string in strings:
    pattern += string + "|"
pattern = pattern[:-1]

class FeatureTBlite:
    def __init__(self) -> None:
        return

    def ImportFeature(self):

        # Use ase to read in coordinates

        faulthandler.enable()

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


            calc= Calculator(
            method="GFN2-xTB",
            uhf=uhf,
            charge=charge,
            numbers=ase_mol.get_atomic_numbers(),
            positions=ase_mol.get_positions()*1/Bohr,
            )

            if self.solvent is not None:
                calc.add("alpb-solvation",self.solvent)

            calc.add("xtbml_xyz")

            res = calc.singlepoint()

            self.gradient = res.get("gradient")

            X = res.get("post-processing-dict")

            self.ml_feat = pd.DataFrame(X, columns=X.keys())

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

                self.dipm[f"delta_{orb}"] = self.ml_feat.loc[:,self.ml_feat.columns.str.contains(f"delta_dipm_{orb}_._")].to_numpy()

                self.qm[f"delta_{orb}"] = self.ml_feat.loc[:,self.ml_feat.columns.str.contains(f"delta_qm_{orb}_.._")].to_numpy()

                if orb == "Z":
                    self.dipm[f"delta_{orb}"] -= self.dipm[f"delta_{orb}"]
                    self.qm[f"delta_{orb}"] -= self.qm[f"delta_{orb}"]

            if orb not in {"e", "Z"}:
                self.dipm[f"{orb}"] = self.ml_feat.loc[:,self.ml_feat.columns.str.startswith(f"dipm_{orb}_")].to_numpy()
                self.qm[f"{orb}"] = self.ml_feat.loc[:,self.ml_feat.columns.str.startswith(f"qm_{orb}_")].to_numpy()

                if orb != "A":
                    self.p[f"{orb}"] = self.ml_feat.loc[:,self.ml_feat.columns.str.contains(f"p_{orb}")].to_numpy()

        self.energy_based = self.ml_feat.loc[:,self.ml_feat.columns.str.contains(pattern)].to_numpy()

        self.cn["default"] = self.ml_feat.loc[:, "CN"].to_numpy()
        self.cn["delta"] = self.ml_feat.loc[:, self.ml_feat.columns.str.contains("delta_CN")].to_numpy()

        self.q["default"] = self.ml_feat.loc[:, "q_A"].to_numpy()
        self.q["delta"] = self.ml_feat.loc[:, self.ml_feat.columns.str.contains("delta_q_A")].to_numpy()
