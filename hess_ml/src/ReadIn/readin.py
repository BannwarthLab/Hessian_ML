import pandas as pd
import numpy as np
import hess_ml.src.constants.constants as const
class ReadXYZ:

    def ImportStructure(self):
        with open(self.xyz_file) as myfile:
            head = [next(myfile) for _ in range(2)]

        xyz_pd = pd.read_csv(
            self.xyz_file,
            sep="\s+",
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
        
        return 