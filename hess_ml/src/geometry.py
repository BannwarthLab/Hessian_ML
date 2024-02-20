from __future__ import annotations

import os
import numpy as np
import pandas as pd

import hess_ml.src.constants.constants as const

class Geometry:
    def __init__(self) -> None:
        return

    def importXYZ(self, file: str):
        with open(file) as myfile:
            [next(myfile) for _ in range(2)]

        xyz_pd = pd.read_csv(
            file,
            sep=r"\s+",
            skiprows=2,
            header=None,
            keep_default_na=False,
            na_values=["_"],
        )

        xyz_pd.columns = ["atoms", "x", "y", "z"]

        self.elements = xyz_pd["atoms"]

        self.N_atoms = len(self.elements)

        self.xyz = np.array(xyz_pd.iloc[:, 1:])

        self.NuclearCharge = np.zeros(self.N_atoms)

        for i in range(self.N_atoms):
            self.NuclearCharge[i] = const.ELEMENTS2Z[self.elements[i].lower()]

        if self.N_atoms == 1:
            self.do_calc = False
            print("At least two atoms must be considered.")

        print(file)

    def importTarget(self, file: str):
        if os.path.isfile(file):
            LineList = []

            with open(file) as fd:
                Lines = [line.rstrip("\n") for line in fd]
                for line in Lines[1:]:
                    LineList += line.split()

            self.target = np.zeros([self.N_atoms * 3, self.N_atoms * 3])

            i = 0

            for j in range(self.N_atoms * 3):
                for k in range(self.N_atoms * 3):
                    self.target[j, k] = float(LineList[i])
                    i += 1
        else:
            self.do_calc = False
