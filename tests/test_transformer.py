import os
import pytest
from mlhess.utils.chemistry.molecule import Molecule
from mlhess.calculator.tblite import Calculator
from mlhess.machinelearning.transform.transform_mlh import transform_training
from mlhess.utils.math.geometrical import get_atom_pair_rot_mat
from mlhess.utils.math.geometrical import supporting_vector
from pathlib import Path
import numpy as np


@pytest.mark.parametrize(
    "fname, atom_pair",
    [
        ("single_mol_data/0002", (0, 1)),
        ("single_mol_data/0002", (1, 2)),
        ("single_mol_data/0002", (0, 2)),
    ],
)
def test_transform_training(fname, atom_pair):
    full_path = os.path.join(Path(__file__).parent, fname)
    mol = Molecule(full_path, "coord.xyz")
    mol.read_hessian("hessian")
    a, b = atom_pair
    h_expected = mol.hessian.hessian[a * 3 : a * 3 + 3, b * 3 : b * 3 + 3]

    calc = Calculator(mol)
    calc.compute_feature()
    transform_training(mol)

    sup_vector = supporting_vector(mol, atom_pair)

    rmat = get_atom_pair_rot_mat(mol.xyz, sup_vector, atom_pair)

    temp_dist_mat = mol.feature.distance_mat.copy()
    temp_dist_mat[np.tril_indices_from(temp_dist_mat)] = np.inf
    atom_pairs = np.argwhere(temp_dist_mat < 20)

    for idx in range(len(atom_pairs)):
        if tuple(atom_pairs[idx]) == atom_pair:
            break

    h = np.array(mol.processed_target[idx]).reshape(3, 3)

    if atom_pair in mol.transpose_list:
        h = h.T

    h_actual = np.matmul(np.matmul(rmat.T, h), rmat)

    np.testing.assert_array_almost_equal(h_actual, h_expected, 5)
