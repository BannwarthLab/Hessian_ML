from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import hess_ml.src.constants.constants as const

if TYPE_CHECKING:
    from hess_ml.src.template import TestMLHessianGFN2xTB

class ReadXYZ:
    def ImportStructure(self:TestMLHessianGFN2xTB):
        with open(self.xyz_file) as myfile:
            [next(myfile) for _ in range(2)]

        xyz_pd = pd.read_csv(
            self.xyz_file,
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

        if self.N_atoms == 1 or self.N_atoms > 400:
            self.do_calc = False
            print("At least two atoms must be considered.")
