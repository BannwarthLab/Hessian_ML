import os
from pathlib import Path

import numpy as np
import pytest

from mlhess.calculator.tblite_wrapper import Calculator
from mlhess.machinelearning.feature.processing_schemes import (
    gen_pair_features,
    gen_pair_features_batched,
    new_gen_pair_features,
)
from mlhess.machinelearning.transform.transform_mlh import (
    collect_feature_arrays,
    rotate_hessian_blocks,
)
from mlhess.utils.chemistry.molecule import Molecule
from mlhess.utils.math.geometrical import get_atom_pair_rot_mat, supporting_vector
from mlhess.utils.math.matrix_operation import get_rotated_33_block_matrix

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
    cwd = os.path.abspath("./")
    os.chdir(os.path.join(Path(__file__).parent, "general_test_dir"))

    mol = Molecule(full_path, "xtbopt.xyz")
    mol.read_hessian("hessian")
    calc = Calculator(mol)
    calc.compute_feature()

    features = []

    for ap, et in zip(atom_pairs, expected_transposes):
        sup_vector = supporting_vector(mol.feature.dipm["A"], ap)
        rmat = get_atom_pair_rot_mat(mol.xyz, sup_vector, ap)

        feature, transpose, rmat_adapted = gen_pair_features(mol.feature, rmat, ap)
        features.append(feature)

        assert transpose == et

    f1 = features[0]
    f2 = features[1]
    np.testing.assert_almost_equal(f1, f2, decimal=5)

    os.chdir(cwd)


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
    cwd = os.path.abspath("./")
    os.chdir(os.path.join(Path(__file__).parent, "general_test_dir"))
    calc = Calculator(mol)
    calc.compute_feature()

    sup_vector = supporting_vector(mol.feature.dipm["A"], atom_pair)
    rmat = get_atom_pair_rot_mat(mol.xyz, sup_vector, atom_pair)

    feature, transpose, rmat_adapted = gen_pair_features(mol.feature, rmat, atom_pair)

    assert transpose == expected_transpose
    os.chdir(cwd)


@pytest.mark.parametrize(
    "fname, xyz_file",
    [
        ("single_mol_data/0001", "xtbopt.xyz"),
        ("single_mol_data/0002", "coord.xyz"),  # linear: degenerate frame
        ("single_mol_data/0003", "xtbopt.xyz"),
        ("single_mol_data/0004", "xtbopt.xyz"),  # same elements, different dipoles
    ],
)
def test_gen_pair_features_batched_equals_per_pair(fname, xyz_file, monkeypatch):
    """Batched features, rotations, swaps and Hessian blocks == per-pair versions."""
    monkeypatch.chdir(os.path.join(Path(__file__).parent, "general_test_dir"))

    mol = Molecule(os.path.join(Path(__file__).parent, fname), xyz_file)
    mol.read_hessian("hessian")
    Calculator(mol).compute_feature()

    arrays = collect_feature_arrays(mol)
    A, B = np.triu_indices(mol.nat, 1)

    features, R, _, _, swap = gen_pair_features_batched(A, B, arrays)
    blocks = rotate_hessian_blocks(mol.hessian.hessian, R, A, B, swap)

    for p, (a, b) in enumerate(zip(A, B)):
        feature_ref, transpose_ref, R_ref = new_gen_pair_features((a, b), arrays)
        block_ref = get_rotated_33_block_matrix(
            R_ref, mol.hessian.hessian, a, b, transpose_ref
        )

        np.testing.assert_allclose(features[p], feature_ref, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(R[p], R_ref, rtol=0, atol=1e-14)
        np.testing.assert_allclose(blocks[p], block_ref.flatten(), rtol=0, atol=1e-14)
        assert swap[p] == (transpose_ref is not None)
