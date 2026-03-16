import os

import pytest
import numpy as np
from pathlib import Path
from mlhess.utils.chemistry.molecule import Molecule
from mlhess.utils.io.reader import read_xyz
import io
from contextlib import redirect_stdout

os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libgomp.so.1"

mol_0001_0 = (np.array([0.0, 0.0, 1.0]), True)
mol_0001_1 = (np.array([0.0, 0.0, 1.0]), False)


@pytest.mark.parametrize(
    "fname, data", [("utils_data/0001", mol_0001_0), ("utils_data/0001", mol_0001_1)]
)
def test_molecule_class(fname, data):
    com_expected, shift = data
    full_path = os.path.join(Path(__file__).parent, fname)
    mol = Molecule(full_path, "coord.xyz")

    el, xyz = read_xyz(os.path.join(full_path, "coord.xyz"))

    com_actual = mol.nuclear_properties.calc_center_of_mass()

    np.testing.assert_almost_equal(com_actual, com_expected)

    mol.nuclear_properties.calc_properties(shift)
    np.testing.assert_almost_equal(mol.nuclear_properties.center_of_mass, com_expected)

    assert mol.nuclear_properties.islin is True

    if shift:
        np.testing.assert_almost_equal(xyz - com_actual, mol.xyz)
    else:
        np.testing.assert_almost_equal(xyz, mol.xyz)

    f = io.StringIO()
    with redirect_stdout(f):
        mol.nuclear_properties.print_center_of_mass()
    s = f.getvalue()

    msg = "Center of mass is at:\n"+"x:  0.0 \n"+"\n"+"y:  0.0 \n"+"\n"+"z:  1.0 \n"+"\n"
    assert s == msg
 