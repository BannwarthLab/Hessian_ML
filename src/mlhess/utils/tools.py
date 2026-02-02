import os
import numpy as np
from mlhess.utils.patcher import patch_molecule
from mlhess.management.config import Configurator
from mlhess.utils.chemistry.molecule import Molecule
from tcgm_lib.IO.writer import wrt_hess_to_xtb


def predict_array(model, folders, config=None):
    if config is None:
        config = Configurator(None)

    with patch_molecule(config):
        for folder in folders:
            mol = Molecule(folder, config.molecule.xyz_file)
            mol.prepare_prediction()
            mol.predict(model)
            if mol.calc_succeeded:
                wrt_hess_to_xtb(
                    os.path.join(mol.path, "MLhessian"), mol.ml_hessian.hessian
                )
                # np.save(os.path.join(mol.path, "MLhessian.npy"),mol.ml_hessian.hessian)

            if mol.nat == 1:
                mol.ml_hessian.hessian = np.zeros([3, 3])
                wrt_hess_to_xtb(
                    os.path.join(mol.path, "MLhessian"), mol.ml_hessian.hessian
                )


def test_array(model, folders, config=None):
    if config is None:
        config = Configurator(None)
    n_val = 0
    err = 0.0
    s_err = 0.0
    with patch_molecule(config):
        for folder in folders:
            mol = Molecule(folder, config.molecule.xyz_file)
            mol.prepare_prediction()
            mol.predict(model)
            mol.read_hessian(config.molecule.target_file)
            err += np.sum(np.abs(mol.ml_hessian.hessian - mol.hessian.hessian))
            s_err += np.sum((mol.ml_hessian.hessian - mol.hessian.hessian) ** 2)
            n_val += mol.hessian.hessian.shape[0] * mol.hessian.hessian.shape[1]

    return np.sqrt(s_err / n_val), err / n_val
