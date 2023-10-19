import os
import unittest as ut
from pathlib import Path

import numpy as np

from hess_ml.src.geometry import Geometry
from hess_ml.src.io import Input


class TestGeometry(ut.TestCase):

    def test_Geometry_data_generation(self) -> None:

        config = {}

        config["xyz_file"] =   "struc.xyz"
        config["feature"] = "tblite"

        folder = Path(__file__).parent / "test_files/ethane/"

        molecule = Geometry(folder = folder,config=config)

        input = Input()

        file = Path(__file__).parent / "test_files/ethane/struc.xyz"

        xyz, header = input.import_coord(file)

        molecule.gen_data(threads=1)

        elements = ["c","c","h","h","h","h","h","h"]
        nuc_charge = [6.0,6.0,1.0,1.0,1.0,1.0,1.0,1.0]

        for i in range(len(elements)):
            assert molecule.elements[i].lower() == elements[i]
            assert molecule.nuc_charge[i] == nuc_charge[i]

        assert molecule.N_atoms == 8
        assert np.sum(np.array(xyz.iloc[:, 1:])) == np.sum(molecule.xyz)


        R = molecule.R_MI_APF_mat[:3,3:6]
        xyz_rot = molecule.xyz[0] - (molecule.xyz[0]+molecule.xyz[1])/2
        xyz_rot = np.matmul(R,xyz_rot)
        self.assertAlmostEqual(xyz_rot[0],0.0,10)
        self.assertAlmostEqual(xyz_rot[1],0.0,10)



if __name__ == "__main___":
    ut.main()
