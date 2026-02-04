import os
import pytest
import numpy as np
from pathlib import Path
from mlhess.machinelearning.feature.base_class import Feature
from mlhess.calculator.tblite_wrapper import Calculator
from mlhess.utils.chemistry.molecule import Molecule

feature = np.float32(
    np.loadtxt(os.path.join(Path(__file__).parent, "single_mol_data/0002/feature.txt"))
)
target = np.float32(
    np.loadtxt(os.path.join(Path(__file__).parent, "single_mol_data/0002/target.txt"))
)


@pytest.mark.parametrize("fname, feature_class", [("single_mol_data/0002", Feature)])
def test_feature(fname, feature_class):
    full_path = os.path.join(Path(__file__).parent, fname)
    mol = Molecule(full_path, "coord.xyz")
    mol.read_hessian("hessian")
    mol.feature_class = feature_class
    calc = Calculator(mol)
    calc.compute_feature()

    assert mol.feature.scalars.shape[1] == 25
    assert mol.feature.vectors.shape[1] == 5
    assert mol.feature.matrices.shape[1] == 4
