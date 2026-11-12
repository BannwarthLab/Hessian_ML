from __future__ import annotations

import numpy as np

from mlhess.utils.decorator import checkTiming


def angle_two_vec(a, b, norm_a, norm_b):
    cosangle = 0 if norm_a == 0 or norm_b == 0 else np.dot(a, b) / (norm_a * norm_b)
    return np.arccos(np.clip(cosangle, -1, 1))


def check_x_axis_error(vec_x, i, j, coord_th=1e-7):
    if np.abs(vec_x[1]) > coord_th or np.abs(vec_x[2]) > coord_th:
        print(f"Error in vec_x for {i, j}")
        print(vec_x)


def check_xyz_error(xyz_i, xyz_j, i, j, rotation_matrix, coord_th=1e-7):
    xyz_temp_i = np.matmul(rotation_matrix, xyz_i)
    xyz_temp_j = np.matmul(rotation_matrix, xyz_j)

    if xyz_temp_i[0] > coord_th or xyz_temp_i[1] > coord_th:
        print(coord_th)
        print(f"Error in coord i:{i, j}")
        print(xyz_temp_i)

    if xyz_temp_j[0] > coord_th or xyz_temp_j[1] > coord_th:
        print(coord_th)

        print(f"Error in coord j:{i, j}")
        print(xyz_temp_j)


# ____Uses for i=j the mean of the xyz's atoms as an artificial atom____
@checkTiming(enabled=False)
def get_atom_pair_rot_mat(xyz: np.ndarray, sup: np.ndarray, indices: tuple[int, int]):
    A, B = indices
    d   = 0.5 * (xyz[A] - xyz[B])                     # A relative to midpoint
    ez  = d / np.linalg.norm(d)
    vx = np.cross(d, sup)
    ex = vx / np.linalg.norm(vx)
    ey = np.cross(ez, ex)
    R   = np.stack([ex, ey, ez], axis=0)              # (P,3,3)
    return R


def supporting_vector_batched(
    vector: np.ndarray, A: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """Batched `supporting_vector` for index arrays A, B of shape (P,).

    Returns:
        np.ndarray: supporting vectors, (P, 3).
    """
    support_vec = vector[A] + vector[B]  # fancy indexing -> new array, (P, 3)
    small = np.abs(support_vec).sum(axis=1) / 3 < 1e-5
    support_vec[small] = (0.0, 1.0, 0.0)
    return support_vec


def get_atom_pair_rot_mat_batched(
    xyz: np.ndarray, sup: np.ndarray, A: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """Batched `get_atom_pair_rot_mat`: rotation matrices for all pairs at once.

    The rows of each matrix are the axes of the atom-pair frame:
        z: from the pair midpoint towards atom A
        x: along (A - midpoint) x supporting vector
        y: z x x (right-handed)

    Args:
        xyz (np.ndarray): coordinates, (nat, 3).
        sup (np.ndarray): supporting vectors, (P, 3).
        A, B (np.ndarray): atom indices, (P,).

    Returns:
        np.ndarray: rotation matrices, (P, 3, 3).
    """
    d = 0.5 * (xyz[A] - xyz[B])  # A relative to the midpoint, (P, 3)
    d_norm = np.linalg.norm(d, axis=1, keepdims=True)
    ez = d / d_norm

    vx = np.cross(d, sup)
    vx_norm = np.linalg.norm(vx, axis=1, keepdims=True)

    # Supporting vector (anti)parallel to the bond, e.g. linear molecules: the x axis
    # is then set by numerical noise in the dipoles, exactly as in the per-pair version
    # (kept for compatibility with trained models). Only if the cross product is
    # exactly zero (per-pair version: NaN) fall back to the coordinate axis least
    # aligned with the bond.
    degenerate = vx_norm[:, 0] == 0.0
    if np.any(degenerate):
        helper = np.eye(3)[np.argmin(np.abs(ez[degenerate]), axis=1)]
        vx[degenerate] = np.cross(ez[degenerate], helper)
        vx_norm[degenerate] = np.linalg.norm(vx[degenerate], axis=1, keepdims=True)

    ex = vx / vx_norm
    ey = np.cross(ez, ex)
    return np.stack([ex, ey, ez], axis=1)  # stack (P,3) rows along axis 1 -> (P,3,3)


def rot_Z(alpha: float) -> np.ndarray:
    """Givens rotation around the z-axis.

    Args:
        alpha (float): Angle in rad.

    Returns:
        np.ndarray: rotation matrix.
    """
    return np.array(
        [
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1],
        ],
    )


def rot_X(alpha: float) -> np.ndarray:
    """Givens rotation around the x-axis.

    Args:
        alpha (float): Angle in rad.

    Returns:
        np.ndarray: rotation matrix.
    """
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)],
        ],
    )


def rot_Y(alpha: float) -> np.ndarray:
    """Givens rotation around the y-axis.

    Args:
        alpha (float): Angle in rad.

    Returns:
        np.ndarray: rotation matrix.
    """

    return np.array(
        [
            [np.cos(alpha), 0.0, -np.sin(alpha)],
            [0.0, 1, 0.0],
            [np.sin(alpha), 0.0, np.cos(alpha)],
        ],
    )


def supporting_vector(vector, atom_pair: tuple[int, int]) -> np.ndarray:
    """Compute supporting vector for the calculation of the rotation matrix."""
    i, j = atom_pair

    support_vec = vector[i] + vector[j]

    # support_vec = np.cross(self._mol.feature.dipm["A"][i],self._mol.feature.dipm["A"][j])

    if np.sum(np.abs(support_vec)) / 3 < 1e-5:
        support_vec = np.array([0.0, 1.0, 0.0])

    return support_vec


# def supporting_vector(mol: Molecule, atom_pair:tuple[int,int]) -> np.ndarray:
#     "Compute supporting vector for the calculation of the rotation matrix."
#     i, j = atom_pair

#     support_vec = mol.feature.dipm["A"][i] + mol.feature.dipm["A"][j]

#     # support_vec = np.cross(self._mol.feature.dipm["A"][i],self._mol.feature.dipm["A"][j])

#     if np.sum(np.abs(support_vec)) / 3 < 1e-5:
#         support_vec = np.array([0.0, 1.0, 0.0])

#     return support_vec
