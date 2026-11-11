import os
import sys
from pathlib import Path

import numpy as np
import pytest

from mlhess.calculator.tblite_wrapper import Calculator
from mlhess.machinelearning.transform.transform_mlh import transform_training
from mlhess.utils.chemistry.molecule import Molecule
from mlhess.utils.math.geometrical import get_atom_pair_rot_mat, supporting_vector


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
    cwd = os.path.abspath("./")
    os.chdir(os.path.join(Path(__file__).parent, "general_test_dir"))
    a, b = atom_pair
    h_expected = mol.hessian.hessian[a * 3 : a * 3 + 3, b * 3 : b * 3 + 3]
    
    num_threads = 1 if sys.platform =='darwin' else 4

    calc = Calculator(mol)
    calc.compute_feature()
    transform_training(mol,num_threads)

    sup_vector = supporting_vector(mol.feature.dipm["A"], atom_pair)

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

    os.chdir(cwd)


@pytest.mark.parametrize(
    "fname, xyz_file",
    [
        ("single_mol_data/0002", "coord.xyz"),
        ("single_mol_data/0004", "xtbopt.xyz"),
    ],
)
def test_hessian_round_trip(fname, xyz_file, monkeypatch):
    """Training targets fed back into the reconstruction give the original off-diagonal blocks."""
    monkeypatch.chdir(os.path.join(Path(__file__).parent, "general_test_dir"))
    full_path = os.path.join(Path(__file__).parent, fname)

    mol_train = Molecule(full_path, xyz_file)
    mol_train.read_hessian("hessian")
    mol_train.prepare_training(num_threads=1)

    mol_pred = Molecule(full_path, xyz_file)
    mol_pred.prepare_prediction(num_threads=1)
    mol_pred.target = np.array(mol_train.processed_target)

    H_rec = mol_pred.ml_hessian.get_processed_target()
    H_ref = mol_train.hessian.hessian
    nat = mol_pred.nat

    np.testing.assert_allclose(H_rec, H_rec.T, atol=1e-12)

    damping = mol_pred.ml_hessian.damping(mol_pred.feature.distance_mat)
    for a in range(nat):
        for b in range(nat):
            if a == b:
                continue
            block_rec = H_rec[3 * a : 3 * a + 3, 3 * b : 3 * b + 3]
            block_ref = H_ref[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] * damping[a, b]
            np.testing.assert_allclose(block_rec, block_ref, atol=1e-10)
