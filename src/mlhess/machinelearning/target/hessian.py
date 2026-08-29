import numpy as np

#from typing import TYPE_CHECKING
from tcgm_lib.molecule.nuclear_hessian import NuclearHessian as AbstractNuclearHessian

from mlhess.utils.math.matrix_operation import (
    get_rotated_33_block_matrix,
    restructure_hessian_rotation_mat,
)

# if TYPE_CHECKING:
#     from mlhess.utils.chemistry.molecule import Molecule

class NuclearHessian(AbstractNuclearHessian):
    def __init__(self, mol):
        super().__init__(mol)
        self._mol = mol

    def apply_adaptation(self):
        pass

    def get_rotated_matrix(self, rot_mat, atom_A, atom_B, transpose):
        return get_rotated_33_block_matrix(rot_mat,self.hessian,atom_A,atom_B,transpose)

    def damping(self, dist):
        return 1 / (np.exp((dist - 13.0) / 0.1) + 1)

    def gen_hess_from_vec_pred_damped(self):
        hess_vec_ab = self._mol.target
        R_MI_APF_mat = self._mol.R_MI_APF_mat
        transpose_list = self._mol.transpose_list
        atom_pairs = np.array(self._mol.computed_atom_pairs)
        dist_mat = self._mol.feature.distance_mat

        # Convert transpose_list to a set for faster membership checking
        transpose_set = set(map(tuple, transpose_list))

        # Use vectorized comparison to check if each (atom_A, atom_B) pair is in transpose_set
        transposes = np.array([tuple(pair) in transpose_set for pair in atom_pairs])

        rabs, hess_ab = restructure_hessian_rotation_mat(
            np.array(hess_vec_ab), atom_pairs, R_MI_APF_mat, np.array(transposes)
        )
        
        hess_ab = np.einsum("mij,mjk,mlk->mil", rabs, hess_ab, rabs)

        Hessian = np.zeros([self._mol.nat * 3, self._mol.nat * 3])

        for ite_hetero, (atom_A, atom_B) in enumerate(zip(atom_pairs[:, 0], atom_pairs[:, 1])):
            Hessian[
                3 * atom_A : 3 * atom_A + 3,
                3 * atom_B : 3 * atom_B + 3,
            ] = hess_ab[ite_hetero] * self.damping(dist_mat[atom_A, atom_B])

            Hessian[
                3 * atom_B : 3 * atom_B + 3,
                3 * atom_A : 3 * atom_A + 3,
            ] = Hessian[
                3 * atom_A : 3 * atom_A + 3,
                3 * atom_B : 3 * atom_B + 3,
            ].T

        for atom_A in range(self._mol.nat):
            for atom_B in range(self._mol.nat):
                if atom_A != atom_B:
                    Hessian[
                        3 * atom_A : 3 * atom_A + 3,
                        3 * atom_A : 3 * atom_A + 3,
                    ] -= Hessian[
                        3 * atom_A : 3 * atom_A + 3,
                        3 * atom_B : 3 * atom_B + 3,
                    ]

            Hessian[3 * atom_A : 3 * atom_A + 3, 3 * atom_A : 3 * atom_A + 3] = (
                Hessian[3 * atom_A : 3 * atom_A + 3, 3 * atom_A : 3 * atom_A + 3]
                + np.transpose(
                    Hessian[3 * atom_A : 3 * atom_A + 3, 3 * atom_A : 3 * atom_A + 3],
                )
            ) / 2

        return Hessian

    def get_processed_target(self):
        self.hessian = self.gen_hess_from_vec_pred_damped()
        return self.hessian


class NuclearHessianPM(NuclearHessian):
    def __init__(self, mol):
        super().__init__(mol)
        self._mol = mol

    def apply_adaptation(self):
        eigvals, eigvecs = np.linalg.eigh(self.hessian)

        self.hess_p = np.zeros_like(self.hessian)
        self.hess_m = np.zeros_like(self.hessian)

        for idx, eigval in enumerate(eigvals):
            if eigval < 0.0:
                self.hess_m += np.outer(eigvecs[:, idx], eigvecs[:, idx]) * eigval
            else:
                self.hess_p += np.outer(eigvecs[:, idx], eigvecs[:, idx]) * eigval

    def get_rotated_matrix(
        self, rot_mat: np.ndarray, atom_A: int, atom_B: int, transpose: bool
    ):
        H_APF_m = get_rotated_33_block_matrix(rot_mat,self.hess_m,atom_A,atom_B,transpose)
        H_APF_p = get_rotated_33_block_matrix(rot_mat,self.hess_p,atom_A,atom_B,transpose)
        return np.array([H_APF_p, H_APF_m])

    def get_processed_target(self):
        upper_tri_blocks_hessian = self._mol.target
        upper_tri_blocks_hessian[:,:9] += upper_tri_blocks_hessian[:,9:]
        self._mol.target = upper_tri_blocks_hessian[:,:9]
        self.hessian = self.gen_hess_from_vec_pred_damped()
        return self.hessian