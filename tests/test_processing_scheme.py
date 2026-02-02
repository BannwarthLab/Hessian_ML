import os

import pytest
import numpy as np
from pathlib import Path
from mlhess.utils.chemistry.molecule import Molecule
from mlhess.calculator.tblite import Calculator
from mlhess.machinelearning.feature.processing_schemes import gen_pair_features
from mlhess.utils.math.geometrical import get_atom_pair_rot_mat, supporting_vector

feature = np.float32(
    np.loadtxt(os.path.join(Path(__file__).parent, "single_mol_data/0002/feature.txt"))
)
target = np.float32(
    np.loadtxt(os.path.join(Path(__file__).parent, "single_mol_data/0002/target.txt"))
)


@pytest.mark.parametrize(
    "fname, atom_pairs, expected_transposes",
    [
        ("single_mol_data/0001", [(0, 1), (0, 2)], (None, None)),
        ("single_mol_data/0003", [(1, 2), (1, 3)], (None, None)),
        ("single_mol_data/0003", [(0, 1), (1, 2)], ([0, 1], None)),
    ],
)
def test_gen_pair_feature_equivalence(fname, atom_pairs, expected_transposes):
    full_path = os.path.join(Path(__file__).parent, fname)
    mol = Molecule(full_path, "xtbopt.xyz")
    mol.read_hessian("hessian")
    calc = Calculator(mol)
    calc.compute_feature()

    features = []

    for ap, et in zip(atom_pairs, expected_transposes):
        sup_vector = supporting_vector(mol, ap)
        rmat = get_atom_pair_rot_mat(mol.xyz, sup_vector, ap)

        feature, transpose, rmat_adapted = gen_pair_features(mol.feature, rmat, ap)
        features.append(feature)

        assert transpose == et

    f1 = features[0]
    f2 = features[1]
    np.testing.assert_almost_equal(f1, f2, decimal=5)


@pytest.mark.parametrize(
    "fname, atom_pair, expected_transpose",
    [
        ("single_mol_data/0004", (2, 5), ([2, 5])),
        ("single_mol_data/0004", (2, 4), ([2, 4])),
    ],
)
def test_gen_pair_feature_same_element_diff_dipm(fname, atom_pair, expected_transpose):
    full_path = os.path.join(Path(__file__).parent, fname)
    mol = Molecule(full_path, "xtbopt.xyz")
    mol.read_hessian("hessian")
    calc = Calculator(mol)
    calc.compute_feature()

    sup_vector = supporting_vector(mol, atom_pair)
    rmat = get_atom_pair_rot_mat(mol.xyz, sup_vector, atom_pair)

    feature, transpose, rmat_adapted = gen_pair_features(mol.feature, rmat, atom_pair)

    assert transpose == expected_transpose
