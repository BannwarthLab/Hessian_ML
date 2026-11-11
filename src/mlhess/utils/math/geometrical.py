from __future__ import annotations

import numpy as np
from numpy import linalg

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
