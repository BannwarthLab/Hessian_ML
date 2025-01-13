"""Basic Vector and Matrix operations."""
from typing import Optional

import numpy as np

def rotate_matrix(rotation_matrix:np.ndarray,matrix:np.ndarray)-> np.ndarray:
    """Rotates a matrix that depends on the coordinates in a second order manner.

    Args:
        rotation_matrix (np.ndarray): Rotatiom matrix
        matrix (np.ndarray): Matrix to rotate.

    Returns:
        np.ndarray: rotated matrix
    """
    return np.matmul(np.matmul(rotation_matrix,matrix),np.transpose(rotation_matrix))

def rotate_vector_array(rotation_matrix:np.ndarray,vectors:np.ndarray)-> np.ndarray:
    """Rotate an array of vectors

    Args:
        rotation_matrix (np.ndarray): Rotatiom matrix
        vectors (np.ndarray): Array of vectors

    Returns:
        np.ndarray: array of rotated vectors
    """
    return np.matmul(vectors,np.transpose(rotation_matrix))


def rotate_vector(rotation_matrix:np.ndarray,vector:np.ndarray)-> np.ndarray:
    """Rotata a vector

    Args:
        rotation_matrix (np.ndarray):  Rotatiom matrix
        vector (np.ndarray): vector

    Returns:
        np.ndarray: rotated vector
    """
    return np.matmul(rotation_matrix,vector)


def fill_matrix_block_AB(
    vector:np.ndarray,
    matrix:np.ndarray,
    R_mat:np.ndarray=None,
    A:Optional[int]=None,
    B:Optional[int]=None,
    transpose:bool=False,
    ) -> np.ndarray:
    """Fill a 3 x 3 matrix block with a rotation matrix and a vector.

    Args:
        vector (np.ndarray): 9 x 1 vector
        matrix (np.ndarray): 3nat x 3nat matrix
        R_mat (np.ndarray, optional): rotatiom matrix. Defaults to None.
        A (int, optional): index 1. Defaults to None.
        B (int, optional): index 2. Defaults to None.
        transpose (bool, optional): true if vector corresponds to lower triangular matrix . Defaults to False.

    Returns:
        np.ndarray: 3nat x 3nat matrix with filled matrix block.
    """
    A3 = 3 * A
    B3 = 3 * B

    matrix[A3 : A3 + 3, B3 : B3 + 3] = vector.reshape(3, 3)

    if transpose is True:
        #matrix[A3 : A3 + 3, B3 : B3 + 3] = self.rotate_matrix(self.rot_Z(np.pi/2),matrix[A3 : A3 + 3, B3 : B3 + 3])
        #matrix[A3 : A3 + 3, B3 : B3 + 3] = self.rotate_matrix(self.rot_X(np.pi),matrix[A3 : A3 + 3, B3 : B3 + 3])
        matrix[A3 : A3 + 3, B3 : B3 + 3] = matrix[A3 : A3 + 3, B3 : B3 + 3].T

    matrix[A3 : A3 + 3, B3 : B3 + 3] = rotate_matrix(R_mat[A3 : A3 + 3, B3 : B3 + 3].T,
                                                            matrix[A3 : A3 + 3, B3 : B3 + 3])

    matrix[B3 : B3 + 3, A3 : A3 + 3] = np.transpose(
        matrix[A3 : A3 + 3, B3 : B3 + 3],
    )

    return matrix


