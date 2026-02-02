from __future__ import annotations

import numpy as np
from numpy import linalg

from mlhess.utils.decorator import checkTiming

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from mlhess.utils.chemistry.molecule import Molecule


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
def get_atom_pair_rot_mat(
    xyz: np.ndarray, supporting_vector: np.ndarray, indices: tuple[int, int]
) -> np.ndarray:
    i, j = indices
    xyz_i = xyz[i].copy()
    xyz_j = xyz[j].copy()

    zero_th = np.float64(0.0)
    sum_th = np.float64(1e-12)

    # assert np.sum(np.abs(xyz_j-xyz_i)) != 0.0, "Cannot compute rotation matrix for two atoms on the same position."

    axis = np.identity(3)

    T = 1 / 2 * xyz_i + 1 / 2 * xyz_j

    xyz_i -= T
    xyz_j -= T

    vec_x = np.cross(xyz_i, supporting_vector)

    LL = np.cross(xyz_i, axis[2])

    xyz_i_norm = np.linalg.norm(xyz_i)
    vec_x_norm = np.linalg.norm(vec_x)
    LL_norm = np.linalg.norm(LL)

    if np.sum(np.abs(np.array([xyz_i[:2], xyz_j[:2]]))) < sum_th:
        alpha = 2 * np.pi - angle_two_vec(vec_x, axis[0], vec_x_norm, 1.0)
        beta = angle_two_vec(xyz_i, axis[2], xyz_i_norm, 1.0)
        gamma = 0

        if linalg.det(np.array([axis[0], axis[2], vec_x])) > zero_th:
            alpha = 2 * np.pi - alpha

    else:
        alpha = angle_two_vec(LL, axis[0], LL_norm, 1.0)
        beta = angle_two_vec(xyz_i, axis[2], xyz_i_norm, 1.0)
        gamma = angle_two_vec(LL, vec_x, LL_norm, vec_x_norm)

        # Find right rotation angle
        if linalg.det(np.array([axis[0], axis[2], LL])) < zero_th:
            alpha = 2 * np.pi - alpha

        if linalg.det(np.array([LL, vec_x, xyz_i])) > zero_th:
            gamma = 2 * np.pi - gamma

    R_euler = np.matmul(
        np.matmul(rot_Z(gamma), rot_X(beta)),
        rot_Z(alpha),
    )

    # Rotation by 180 ° if dipole moment is negative in x

    if np.matmul(R_euler, vec_x)[0] < zero_th:
        R_z = rot_Z(np.pi)
        R_euler = np.matmul(R_z, R_euler)

    # Apply the euler rotation matrix on the new x axis for verification reasons
    vec_x = np.matmul(R_euler, vec_x)

    angle_two_vec(LL, vec_x, LL_norm, vec_x_norm)

    # Check for Errors in dipole moment or the coordinates

    if vec_x[0] < zero_th:
        print(f"Error in vec_x[0] in {i, j}")

    check_x_axis_error(vec_x, i, j)

    check_xyz_error(xyz_i, xyz_j, i, j, R_euler, coord_th=1e-7)

    return R_euler


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


def supporting_vector(mol: Molecule, atom_pair):
    i, j = atom_pair

    support_vec = mol.feature.dipm["A"][i] + mol.feature.dipm["A"][j]

    # support_vec = np.cross(self._mol.feature.dipm["A"][i],self._mol.feature.dipm["A"][j])

    if np.sum(np.abs(support_vec)) / 3 < 1e-5:
        support_vec = np.array([0.0, 1.0, 0.0])

    return support_vec
