import faulthandler
import os

import numpy as np
import pandas as pd
from ase.io import read as ase_read
from ase.units import Bohr, Hartree
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

            charge = 0
            uhf = 0

            chargeFilePath = os.path.join(self.folder, ".CHRG")
            if os.path.isfile(chargeFilePath):
                with open(chargeFilePath) as chrg:
                    charge = int(chrg.readline())

            uhfFilePath = os.path.join(self.folder, ".UHF")

            if os.path.isfile(uhfFilePath):
                with open(uhfFilePath) as uhfFile:
                    uhf = int(uhfFile.readline())

            calc= Calculator(
            method="GFN2-xTB",
            uhf=uhf,
            charge=charge,
            numbers=self.NuclearCharge,
            positions=self.xyz*1/Bohr,
            )

            if self.solvent is not None:
                calc.add("alpb-solvation",self.solvent)

            calc.add("xtbml_xyz")

            res = calc.singlepoint()

            self.gradient = res.get("gradient")*1/Bohr

            X = res.get("post-processing-dict")

            self.ml_feat = pd.DataFrame(X, columns=X.keys())

            self.FilterFeatures()

            with open(os.path.join(self.folder,'dftd4.out')) as fname:
                lines = fname.readlines()
            fname.close()

            for j,line in enumerate(lines):
                if "     #    Z              CN          q   " in line:
                    header_idx = j-2

                if "Molecular properties (in atomic units):" in line:
                    footer_idx = len(lines)-j+2

            df = pd.read_csv(os.path.join(self.folder,'dftd4.out'),
                             names=['#','Z','CN','q','C6','C8'],
                             sep='\s+',header=header_idx,skipfooter=footer_idx,
                             engine='python')
            
            self.C6_params = df['C6'].to_numpy()
            self.C8_params = df['C8'].to_numpy()

        except:  # noqa: E722

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

                self.dipm[f"delta_{orb}"] = self.ml_feat.loc[:,
                    self.ml_feat.columns.str.contains(f"delta_dipm_{orb}_._")].to_numpy()

                self.qm[f"delta_{orb}"] = self.ml_feat.loc[:,
                    self.ml_feat.columns.str.contains(f"delta_qm_{orb}_.._")].to_numpy()

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
