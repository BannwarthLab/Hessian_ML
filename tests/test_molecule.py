import os
import pytest
import numpy as np
from pathlib import Path
from mlhess.utils.chemistry.molecule import Molecule
from mlhess.machinelearning.target.hessian import NuclearHessianPM

os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libgomp.so.1"


mol_0001 = {
    "xyz": tuple([2, np.array([0.63711652837626, 1.01749072177456, 0.00407739498116])]),
    "nat": 5,
}

feature = np.float32(
    np.loadtxt(os.path.join(Path(__file__).parent, "single_mol_data/0002/feature.txt"))
)
target = np.float32(
    np.loadtxt(os.path.join(Path(__file__).parent, "single_mol_data/0002/target.txt"))
)


@pytest.mark.parametrize(
    "fname, data, hess_class",
    [
        ("single_mol_data/0002", (feature, target), "mlh"),
        ("single_mol_data/0002", (feature, target), "mlh_pm"),
    ],
)
def test_prepare_training(fname, data, hess_class):
    feature, target = data
    full_path = os.path.join(Path(__file__).parent, fname)
    mol = Molecule(full_path, "coord.xyz")
    if hess_class == "mlh_pm":
        mol.hess_class = NuclearHessianPM
    cwd = os.path.abspath("./")
    os.chdir(os.path.join(Path(__file__).parent,'general_test_dir'))

    mol.read_hessian("hessian")
    mol.prepare_training()

    test_target = np.float32(mol.processed_target)

    if hess_class == "mlh_pm":
        test_target = test_target[:, :9] + test_target[:, 9:]

    test_feature = np.float32(mol.feature.processed)

    np.testing.assert_array_almost_equal(test_target, target, 5)
    np.testing.assert_allclose(
        test_feature[feature > 1e-7], feature[feature > 1e-7], 1e-4
    )
    os.chdir(cwd)


@pytest.mark.parametrize("fname, data", [("single_mol_data/0001", mol_0001)])
def test_molecule_class(fname, data):
    full_path = os.path.join(Path(__file__).parent, fname)
    mol = Molecule(full_path, "xtbopt.xyz")
    assert mol.nat == data["nat"]
    idx, xyz = data["xyz"]
    np.testing.assert_almost_equal(mol.xyz[idx], xyz)

    hess = mol.read_hessian("hessian")
    assert np.abs(np.sum(hess)) < 1e-9
