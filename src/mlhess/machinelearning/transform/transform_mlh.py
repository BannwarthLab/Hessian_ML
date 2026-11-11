"""Transformation of molecules into (feature, target) pairs per atom pair.

All atom pairs of a molecule are processed batch-wise with vectorized numpy
operations (see `gen_pair_features_batched`). The pairs are split into chunks of
`CHUNK_SIZE` to keep the memory bounded for large molecules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mlhess.machinelearning.feature.processing_schemes import gen_pair_features_batched
from mlhess.machinelearning.target.hessian import NuclearHessianPM

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mlhess.utils.chemistry.molecule import Molecule

# Number of atom pairs processed at once (~ CHUNK_SIZE * 369 * 8 B = 60 MB of features).
CHUNK_SIZE = 20_000

# Atom pairs with a distance below this cutoff (in Bohr) are considered.
PAIR_CUTOFF = 20.0


def collect_feature_arrays(mol: Molecule) -> tuple:
    """Collect the molecular arrays needed for the pair features."""
    dipm_key = "A"
    qm_key = "A"

    atomic_numbers = np.array(mol.atomic_numbers)
    dipm_norm = np.array(mol.feature.dipm_norm)
    xyz = mol.xyz
    distance_mat = np.array(mol.feature.distance_mat)
    q = np.array(mol.feature.q["default"])
    dipm = np.array(mol.feature.dipm[dipm_key])
    qm = np.array(mol.feature.qm[qm_key])
    scalars = mol.feature.scalars
    vectors = mol.feature.vectors
    matrices = mol.feature.matrices
    etot_idx = np.array([mol.scalar_keys.index("E_tot")])
    C6_params = mol.feature.C6_params
    wbo = mol.feature.wbo

    return (
        atomic_numbers,
        dipm_norm,
        xyz,
        distance_mat,
        q,
        dipm,
        qm,
        scalars,
        vectors,
        matrices,
        etot_idx,
        C6_params,
        wbo,
    )


def select_atom_pairs(mol: Molecule, cutoff: float = PAIR_CUTOFF) -> np.ndarray:
    """Atom pairs (A < B) closer than `cutoff`, shape (P, 2)."""
    temp_dist_mat = mol.feature.distance_mat.copy()
    temp_dist_mat[np.tril_indices_from(temp_dist_mat)] = np.inf
    return np.argwhere(temp_dist_mat < cutoff)


def _chunks(n: int, size: int = CHUNK_SIZE) -> Iterator[slice]:
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def rotate_hessian_blocks(
    hessian: np.ndarray,
    R: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    swap: np.ndarray,
) -> np.ndarray:
    """Rotated 3x3 Hessian blocks H_AB of all pairs (batched `get_rotated_33_block_matrix`).

    Args:
        hessian (np.ndarray): (3nat, 3nat), or (n, 3nat, 3nat) for several
            Hessians (e.g. H+ and H- for NuclearHessianPM).
        R (np.ndarray): (adapted) rotation matrices, (P, 3, 3).
        A, B (np.ndarray): ORIGINAL atom indices of the pairs (before the canonical
            reordering), (P,). The block is taken at H[3A:3A+3, 3B:3B+3].
        swap (np.ndarray): True where A and B were swapped -> block is transposed, (P,).

    Returns:
        np.ndarray: flattened blocks, (P, 9) or (P, n * 9).
    """
    nat = hessian.shape[-1] // 3
    P = len(A)

    # (n, 3nat, 3nat) -> (n, nat, 3, nat, 3) -> (nat, nat, n, 3, 3) -> pick (P, n, 3, 3)
    H = hessian.reshape(-1, nat, 3, nat, 3).transpose(1, 3, 0, 2, 4)
    blocks = H[A, B]

    Rn = R[:, None]  # broadcast over the n Hessians
    blocks = Rn @ blocks @ Rn.swapaxes(-1, -2)
    blocks[swap] = blocks[swap].swapaxes(-1, -2)

    return blocks.reshape(P, -1)


def _transpose_entries(pairs: np.ndarray, swap: np.ndarray) -> list[list[int]]:
    """Original (A, B) of swapped pairs, the format of `Molecule.transpose_list`."""
    return pairs[swap].tolist()


def transform_prediction(self: Molecule, num_threads: int = 1) -> None:
    """Processes the features to ML readable.

    Args:
        self (Molecule): molecule with computed features.
        num_threads (int): unused, kept for API compatibility (the transformation
            is vectorized instead of parallelized).
    """
    if self.nat == 1:
        self.calc_succeeded = False

    if not self.calc_succeeded:
        return

    pairs = select_atom_pairs(self)
    self.computed_atom_pairs = list(pairs)
    arrays = collect_feature_arrays(self)

    nat = self.nat
    # View of R_MI_APF_mat as (nat, nat, 3, 3): R4[a, b] is the block [3a:3a+3, 3b:3b+3]
    R4 = self.R_MI_APF_mat.reshape(nat, 3, nat, 3).transpose(0, 2, 1, 3)

    features = []
    for chunk in _chunks(len(pairs)):
        A0, B0 = pairs[chunk, 0], pairs[chunk, 1]
        feats, R, _, _, swap = gen_pair_features_batched(A0, B0, arrays)

        features.append(feats)
        self.transpose_list.extend(_transpose_entries(pairs[chunk], swap))
        R4[A0, B0] = R  # stored at the original (A, B) position

    self.feature.processed = np.concatenate(features)


def transform_training(mol: Molecule, num_threads: int = 1) -> None:
    """Transforms the Hessian matrix and features of a molecule into a target-descriptor relation for training.

    Args:
        mol (Molecule): molecule with computed features and Hessian.
        num_threads (int): unused, kept for API compatibility (the transformation
            is vectorized instead of parallelized).
    """
    if mol.nat == 1:
        mol.calc_succeeded = False

    if mol.calc_succeeded:
        mol.transpose_list = []

        pairs = select_atom_pairs(mol)
        mol.computed_atom_pairs = list(pairs)
        arrays = collect_feature_arrays(mol)

        if isinstance(mol.hessian, NuclearHessianPM):
            hessian = np.stack([mol.hessian.hess_p, mol.hessian.hess_m])
        else:
            hessian = mol.hessian.hessian

        features, targets = [], []
        for chunk in _chunks(len(pairs)):
            A0, B0 = pairs[chunk, 0], pairs[chunk, 1]
            feats, R, _, _, swap = gen_pair_features_batched(A0, B0, arrays)

            features.append(feats)
            # block at the original (A0, B0) position, as in the per-pair version
            targets.append(rotate_hessian_blocks(hessian, R, A0, B0, swap))
            mol.transpose_list.extend(_transpose_entries(pairs[chunk], swap))

        mol.feature.processed = np.concatenate(features)
        mol.processed_target = np.concatenate(targets)

    mol.feature.processed = np.array(mol.feature.processed)
