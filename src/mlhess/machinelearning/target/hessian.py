import numpy as np

# from typing import TYPE_CHECKING
from tcgm_lib.molecule.nuclear_hessian import NuclearHessian as AbstractNuclearHessian

from mlhess.utils.math.matrix_operation import get_rotated_33_block_matrix

# if TYPE_CHECKING:
#     from mlhess.utils.chemistry.molecule import Molecule


class NuclearHessian(AbstractNuclearHessian):
    def __init__(self, mol):
        super().__init__(mol)
        self._mol = mol

    def apply_adaptation(self):
        pass

    def get_rotated_matrix(self, rot_mat, atom_A, atom_B, transpose):
        return get_rotated_33_block_matrix(
            rot_mat, self.hessian, atom_A, atom_B, transpose
        )

    def damping(self, dist):
        return 1 / (np.exp((dist - 13.0) / 0.1) + 1)

    def gen_hess_from_vec_pred_damped(self) -> np.ndarray:
        """Build the full Hessian from the predicted 3x3 blocks of all atom pairs.

        Inverse of `transform_mlh.rotate_hessian_blocks`, batched over all pairs:
            1. reshape predicted vectors to 3x3 blocks, transpose swapped pairs
            2. rotate back to the molecular frame: R^T H_APF R
            3. damp by distance and place H_AB and H_BA = H_AB^T
            4. diagonal blocks from the sum rule H_AA = -sum_{B != A} H_AB, symmetrized
        """
        nat = self._mol.nat
        atom_pairs = np.asarray(self._mol.computed_atom_pairs).reshape(-1, 2)
        A, B = atom_pairs[:, 0], atom_pairs[:, 1]
        P = len(A)

        # 1. blocks in the atom-pair frame, (P, 3, 3)
        hess_ab = np.asarray(self._mol.target, dtype=np.float64).reshape(P, 3, 3).copy()

        # transpose_list holds the original (A, B) of swapped pairs -> boolean mask via
        # unique pair ids A * nat + B
        transpose_ids = np.asarray(self._mol.transpose_list, dtype=np.int64).reshape(-1, 2)
        swap = np.isin(A * nat + B, transpose_ids[:, 0] * nat + transpose_ids[:, 1])
        hess_ab[swap] = hess_ab[swap].swapaxes(-1, -2)

        # 2. back-rotation with R = R_MI_APF_mat[3A:3A+3, 3B:3B+3]: R^T H R
        R = self._mol.R_MI_APF_mat.reshape(nat, 3, nat, 3).transpose(0, 2, 1, 3)[A, B]
        hess_ab = R.swapaxes(-1, -2) @ hess_ab @ R

        # 3. damping and placement; H4[a, b] is the block [3a:3a+3, 3b:3b+3]
        hess_ab *= self.damping(self._mol.feature.distance_mat[A, B])[:, None, None]

        H4 = np.zeros((nat, nat, 3, 3))
        H4[A, B] = hess_ab
        H4[B, A] = hess_ab.swapaxes(-1, -2)

        # 4. sum rule (diagonal blocks are still zero here), then symmetrize
        diag = -H4.sum(axis=1)  # (nat, 3, 3)
        idx = np.arange(nat)
        H4[idx, idx] = 0.5 * (diag + diag.swapaxes(-1, -2))

        return H4.transpose(0, 2, 1, 3).reshape(3 * nat, 3 * nat)

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
        H_APF_m = get_rotated_33_block_matrix(
            rot_mat, self.hess_m, atom_A, atom_B, transpose
        )
        H_APF_p = get_rotated_33_block_matrix(
            rot_mat, self.hess_p, atom_A, atom_B, transpose
        )
        return np.array([H_APF_p, H_APF_m])

    def get_processed_target(self):
        upper_tri_blocks_hessian = self._mol.target
        upper_tri_blocks_hessian[:, :9] += upper_tri_blocks_hessian[:, 9:]
        self._mol.target = upper_tri_blocks_hessian[:, :9]
        self.hessian = self.gen_hess_from_vec_pred_damped()
        return self.hessian
