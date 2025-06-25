import os
import unittest as ut
from pathlib import Path

import numpy as np
from hess_ml.src2.molecule.molecule import Molecule
from hess_ml.src2.molecule.tblite.prediction_feature import Feature

class TestGeometry(ut.TestCase):

    def test_Geometry_data_generation(self) -> None:

        config = {}

        fxyz =   "struc.xyz"

        folder = Path(__file__).parent / "test_files/ethane/"

        molecule = Molecule(folder=folder,fxyz=fxyz)
        molecule.feature = Feature
        molecule.read_hessian(fhess='hessian',hesstype='xTB')
        molecule.feature.processed_features

        R = molecule.feature.R_MI_APF_mat
        xyz_rot = molecule.xyz[0] - (molecule.xyz[0]+molecule.xyz[1])/2
        xyz_rot = np.matmul(R,xyz_rot)
        self.assertAlmostEqual(xyz_rot[0],0.0,10)
        self.assertAlmostEqual(xyz_rot[1],0.0,10)

if __name__ == "__main___":
    ut.main()
