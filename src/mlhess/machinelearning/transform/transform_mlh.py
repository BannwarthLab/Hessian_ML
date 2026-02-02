from __future__ import annotations
import numpy as np

from typing import TYPE_CHECKING
from copy import deepcopy

from multiprocessing import Pool

from mlhess.utils.math.geometrical import get_atom_pair_rot_mat
from mlhess.machinelearning.feature.processing_schemes import gen_pair_features
from mlhess.utils.math.geometrical import supporting_vector
import time as time

if TYPE_CHECKING:
    from mlhess.utils.chemistry.molecule import Molecule


def init_worker(mol: Molecule):
    global GLOBAL_MOL
    GLOBAL_MOL = mol  # type: ignore[name-defined]


def transform_prediction(self: Molecule, num_threads):
    """Processes the features to ML readable."""
    if self.nat == 1:
        self.calc_succeeded = False

    if self.calc_succeeded:
        temp_dist_mat = self.feature.distance_mat.copy()
        temp_dist_mat[np.tril_indices_from(temp_dist_mat)] = np.inf
        atom_pairs = np.argwhere(temp_dist_mat < 20)

        self.computed_atom_pairs = atom_pairs

        init_worker(self)
        print(num_threads)
        with Pool(processes=num_threads) as pool:
            results_iterator = pool.map(_transform_block_prediction, atom_pairs)
            pool.terminate()
            pool.join()
        pool.close()

        R_MI_APFs, processed_feature, transposes = zip(*deepcopy(results_iterator))

        self.feature.processed = processed_feature

        transposes = list(transposes)

        for val in transposes:
            if val is not None:
                self.transpose_list.append(val)

        for atom_pair, rot_mat in zip(atom_pairs, R_MI_APFs):
            atom_A, atom_B = atom_pair
            i0 = 3 * atom_A
            i3 = 3 * atom_A + 3
            j0 = 3 * atom_B
            j3 = 3 * atom_B + 3

            self.R_MI_APF_mat[i0:i3, j0:j3] = rot_mat


def transform_training(mol: Molecule, num_threads: int = 4):
    """Transforms the Hessian matrix and features of a molecule into a target-descriptor relation for training."""

    if mol.nat == 1:
        mol.calc_succeeded = False

    if mol.calc_succeeded:
        mol.transpose_list = []

        temp_dist_mat = mol.feature.distance_mat.copy()
        temp_dist_mat[np.tril_indices_from(temp_dist_mat)] = np.inf
        atom_pairs = np.argwhere(temp_dist_mat < 20, dtype=int)

        mol.computed_atom_pairs = atom_pairs

        init_worker(mol)
        with Pool(processes=num_threads) as pool:
            results_iterator = pool.map(_transform_block_training, atom_pairs)
            pool.terminate()
            pool.join()
        pool.close()

        mol.feature.processed, mol.processed_target, transposes = process_results(
            results_iterator
        )

        for val in transposes:
            if val is not None:
                mol.transpose_list.append(val)

        del atom_pairs, pool, results_iterator

    mol.feature.processed = np.array(mol.feature.processed)


def _transform_block_prediction(atom_pair: list[int]) -> tuple:
    """Construction of feature vector for the prediction of the an AB Hessian matrix block.

    Args:
        index (int): index of for the list of atom pairs.
        atom_pairs (list[tuple]): list of atom pairs for the comp. of the rotation matrix.

    Returns:
        tuple: Rotation Matrix, Features, transpose info
    """
    mol = GLOBAL_MOL  # type: ignore[name-defined]
    sup_vector = supporting_vector(mol, atom_pair)

    R_MI_APF = get_atom_pair_rot_mat(mol.xyz, sup_vector, atom_pair)

    Feature_AB, transpose, R_MI_APF = gen_pair_features(
        mol.feature,
        R_MI_APF,
        atom_pair,  # type: ignore[arg-type]
    )

    return R_MI_APF, Feature_AB, transpose


def _transform_block_training(atom_pair: list[int]) -> tuple:
    """Rotation of an AB Hessian matrix block and the construction its feature vector.

    :param index: index of for the list of atom pairs
    :param atom_pairs: list of atom pairs for the comp. of the rotation matrix"""
    mol = GLOBAL_MOL  # type: ignore[name-defined]

    atom_A, atom_B = atom_pair
    sup_vector = supporting_vector(mol, atom_pair)

    R_MI_APF = get_atom_pair_rot_mat(
        mol.xyz,
        sup_vector,
        atom_pair,
    )

    Feature_AB, transpose, R_MI_APF = gen_pair_features(
        mol.feature,
        R_MI_APF,  # type: ignore[arg-type]
        atom_pair,  # type: ignore[arg-type]
    )

    H_APF = mol.hessian.get_rotated_matrix(R_MI_APF, atom_A, atom_B, transpose)

    return list(H_APF.flatten()), Feature_AB, transpose


def process_results(results) -> tuple[list, list, list]:
    # Process and release memory for results incrementally
    processed_features: list[np.array] = []
    processed_target: list[np.array] = []

    processed_target.extend(result[0] for result in results)
    processed_features.extend(result[1] for result in results)
    transposes: list[list | None] = [result[2] for result in results]
    # Release memory for results
    del results

    return processed_features, processed_target, transposes
