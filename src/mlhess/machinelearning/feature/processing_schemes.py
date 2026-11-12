import numpy as np

from mlhess.utils.math.geometrical import (
    get_atom_pair_rot_mat,
    get_atom_pair_rot_mat_batched,
    supporting_vector,
    supporting_vector_batched,
)
from mlhess.utils.math.matrix_operation import rotate_matrix, rotate_vector_array

# if TYPE_CHECKING:
#     from mlhess.machinelearning.feature.base_class import Feature

TRIL = np.tril_indices(3)          # module level, computed once

# Powers of R_AB used as features. The 6 (instead of -6) and the duplicated -12 are
# unintended, but existing models were trained with them -> keep for compatibility.
R_AB_POWERS = np.array([-12, 6, 1, -1, -3, -6, -12])


def gen_pair_features(self, R_MI_APF, atom_pair: tuple[int, int]) -> tuple:
    """Generate features for an atom pair.

    Called with a Feature instance as ``self``, holding the computed
    molecular features.

    Args:
        R_MI_APF (np.ndarray): Rotation matrix.
        atom_pair (tuple[int, int]): Index of atom A and atom B.

    Returns:
        tuple: Features, transpose info, and rotation matrix.
    """
    Features_temp = []

    A, B = atom_pair

    transpose = None

    # Performs a rotation around the Y axis by 180 ° if nuclear charge of A
    # is smaller than B to achieve a consistent alignment
    # If A == B rotation depends on dipole moment

    if self._mol.atomic_numbers[A] < self._mol.atomic_numbers[B]:
        B, A = A, B

        transpose = [B, A]

        # rotation around the y axis
        R_MI_APF = -R_MI_APF
        R_MI_APF[1] = -R_MI_APF[1]

    elif self._mol.atomic_numbers[A] == self._mol.atomic_numbers[B]:
        dipm_A_norm = self.dipm_norm[A]
        dipm_B_norm = self.dipm_norm[B]

        if dipm_A_norm < dipm_B_norm:
            B, A = A, B

            transpose = [B, A]

            # rotation around the y axis
            R_MI_APF = -R_MI_APF
            R_MI_APF[1] = -R_MI_APF[1]

        elif dipm_A_norm == dipm_B_norm:
            print("Nucelar Charge and Dipole moment are the same.")

    xyz_A = self._mol.xyz[A, :].copy()
    xyz_B = self._mol.xyz[B, :].copy()

    s = 0.5 * (xyz_A + xyz_B)

    xyz_A -= s
    xyz_B -= s
    xyz_A = np.matmul(R_MI_APF, xyz_A)
    xyz_B = np.matmul(R_MI_APF, xyz_B)

    r_AB = (xyz_A - xyz_B).reshape(1, -1)

    R_AB = self.distance_mat[A, B]

    Quantity_AB: list[list[float]] = [[], []]
    etot_idx = self.scalar_keys.index("E_tot")
    # Atom specific information

    for j, atom in enumerate([A, B]):
        # ____Rotation from initial coordinate system to atom pair focused system____

        # This is a 7 x 3 x 3 matrix, rotation of all matrices is achieved with this routine

        quad_moments = rotate_matrix(R_MI_APF, self.matrices[atom])

        # Only takes the lower triangle values as the matrices are symmetric
        Quantity_AB[j].extend(
            quad_moments[:, TRIL[0], TRIL[1]]
            .flatten()
            .tolist()
        )

        dipole_moments = rotate_vector_array(R_MI_APF, self.vectors[atom])

        Quantity_AB[j].extend(dipole_moments.flatten().tolist())

        # ____Append Features to Feature Vector____

        Quantity_AB[j].extend(self.scalars[atom])
        Quantity_AB[j].extend(self.scalars[atom] / self.scalars[atom][etot_idx])

    Quantity_AB_arr = np.array(Quantity_AB)

    # Feature_Arith = ((Quantity_AB_arr[0] + Quantity_AB_arr[1]) / 2).tolist()
    Feature_Prod = (Quantity_AB_arr[0] * Quantity_AB_arr[1]).tolist()

    Feature_AbsDiff = (Quantity_AB_arr[0] - Quantity_AB_arr[1]).tolist()

    Features_temp.extend(Quantity_AB[0])
    Features_temp.extend(Quantity_AB[1])

    # atom pair information

    r_BA = -r_AB

    dipm_key = "A"

    dipm_A = self.dipm[dipm_key][A].reshape(1, -1)
    dipm_B = self.dipm[dipm_key][B].reshape(1, -1)

    dipm_A = np.matmul(R_MI_APF, dipm_A.T).T
    dipm_B = np.matmul(R_MI_APF, dipm_B.T).T

    q_A = self.q["default"][A]
    q_B = self.q["default"][B]

    order1_aes = q_A * np.dot(dipm_B, r_BA.T) + q_B * np.dot(dipm_A, r_AB.T)
    order1_aes /= R_AB**3

    qm_key = "A"

    qm_A = self.qm[qm_key][A]
    qm_A = rotate_matrix(R_MI_APF, qm_A)

    qm_B = self.qm[qm_key][B]

    qm_B = rotate_matrix(R_MI_APF, qm_B)

    order2_aes = q_A * np.matmul(r_AB, np.matmul(qm_B, r_AB.T))
    order2_aes += q_B * np.matmul(r_AB, np.matmul(qm_A, r_AB.T))
    order2_aes -= 3 * np.dot(dipm_A, r_AB.T) * np.dot(dipm_B, r_AB.T)
    order2_aes += R_AB**2 * np.dot(dipm_A, dipm_B.T)

    order2_aes /= R_AB**5

    C6_A = float(self.C6_params[A])
    C6_B = float(self.C6_params[B])

    potE = q_A * q_B / R_AB

    # atoms = [A,B]
    # for atom in atoms:
    #     mask = np.ones(self._mol.nat,dtype=bool)
    #     mask[atom] = False
    #     V_J = np.sum(self.q["default"][mask]/self.distance_mat[atom,mask])
    #    Features_temp.append(V_J*self.q["default"][atom])

    Features_temp.append(potE)
    Features_temp.append(C6_A)
    Features_temp.append(C6_B)

    Features_temp.append(self.wbo[A, B])
    Features_temp.extend(order1_aes[0])
    Features_temp.extend(order2_aes[0])

    # Features_temp.extend(Feature_Arith)

    Features_temp.extend(Feature_Prod)
    Features_temp.extend(Feature_AbsDiff)

    Features_temp.extend(R_AB**R_AB_POWERS)

    return np.array(Features_temp), transpose, R_MI_APF


def new_gen_pair_features(atom_pair: tuple[int, int], *args) -> tuple:
    """Generate features for an atom pair.

    Args:
        atom_pair (tuple[int, int]): Index of atom A and atom B.
        *args: Single-element tuple containing the shared molecular arrays
            (atomic_numbers, dipm_norm, xyz, distance_mat, q, dipm, qm,
            scalars, vectors, matrices, etot_idx, C6_params, wbo).

    Returns:
        tuple: Features, transpose info, and rotation matrix.
    """
    Features_temp = []

    (
        atomic_numbers,
        dipm_norm,
        xyz,
        distance_mat,
        q,
        at_dipole,
        qm,
        scalars,
        vectors,
        matrices,
        etot_idx,
        C6_params,
        wbo,
    ) = args[0]

    sup_vector = supporting_vector(at_dipole, atom_pair)

    R_MI_APF = get_atom_pair_rot_mat(xyz, sup_vector, atom_pair)

    R_MI_APF, atom_pair, transpose = adapt_rotation_matrix(
        R_MI_APF, atom_pair, atomic_numbers, dipm_norm
    )

    r_AB, R_AB = compute_rab(xyz, R_MI_APF, atom_pair, distance_mat)

    A, B = atom_pair

    Quantity_AB: list[list[float]] = [[], []]

    # Atom specific information

    for j, atom in enumerate([A, B]):
        # ____Rotation from initial coordinate system to atom pair focused system____

        # This is a 4 x 3 x 3 matrix, rotation of all matrices is achieved with this routine

        quad_moments = rotate_matrix(R_MI_APF, matrices[atom])

        # Only takes the lower triangle values as the matrices are symmetric
        Quantity_AB[j].extend(
            quad_moments[:, TRIL[0], TRIL[1]]
            .flatten()
            .tolist()
        )

        dipole_moments = rotate_vector_array(R_MI_APF, vectors[atom])

        Quantity_AB[j].extend(dipole_moments.flatten().tolist())

        # ____Append Features to Feature Vector____

        Quantity_AB[j].extend(scalars[atom])
        Quantity_AB[j].extend(scalars[atom] / scalars[atom][etot_idx[0]])

    Quantity_AB_arr = np.array(Quantity_AB)

    # Feature_Arith = ((Quantity_AB_arr[0] + Quantity_AB_arr[1]) / 2).tolist()
    Feature_Prod = (Quantity_AB_arr[0] * Quantity_AB_arr[1]).tolist()

    Feature_AbsDiff = (Quantity_AB_arr[0] - Quantity_AB_arr[1]).tolist()

    Features_temp.extend(Quantity_AB[0])
    Features_temp.extend(Quantity_AB[1])

    # atom pair information

    r_BA = -r_AB

    dipm_A = at_dipole[A].reshape(1, -1)
    dipm_B = at_dipole[B].reshape(1, -1)

    dipm_A = np.matmul(R_MI_APF, dipm_A.T).T
    dipm_B = np.matmul(R_MI_APF, dipm_B.T).T

    q_A = q[A]
    q_B = q[B]

    order1_aes = q_A * np.dot(dipm_B, r_BA.T) + q_B * np.dot(dipm_A, r_AB.T)
    order1_aes /= R_AB**3

    qm_A = qm[A]
    qm_A = rotate_matrix(R_MI_APF, qm_A)

    qm_B = qm[B]

    qm_B = rotate_matrix(R_MI_APF, qm_B)

    order2_aes = q_A * np.matmul(r_AB, np.matmul(qm_B, r_AB.T))
    order2_aes += q_B * np.matmul(r_AB, np.matmul(qm_A, r_AB.T))
    order2_aes -= 3 * np.dot(dipm_A, r_AB.T) * np.dot(dipm_B, r_AB.T)
    order2_aes += R_AB**2 * np.dot(dipm_A, dipm_B.T)

    order2_aes /= R_AB**5

    C6_A = float(C6_params[A])
    C6_B = float(C6_params[B])

    potE = q_A * q_B / R_AB

    # atoms = [A,B]
    # for atom in atoms:
    #     mask = np.ones(self._mol.nat,dtype=bool)
    #     mask[atom] = False
    #     V_J = np.sum(self.q["default"][mask]/self.distance_mat[atom,mask])
    #    Features_temp.append(V_J*self.q["default"][atom])

    Features_temp.append(potE)
    Features_temp.append(C6_A)
    Features_temp.append(C6_B)

    Features_temp.append(wbo[A, B])

    Features_temp.extend(order1_aes[0])
    Features_temp.extend(order2_aes[0])

    # Features_temp.extend(Feature_Arith)

    Features_temp.extend(Feature_Prod)
    Features_temp.extend(Feature_AbsDiff)

    Features_temp.extend(R_AB**R_AB_POWERS)

    return np.array(Features_temp), transpose, R_MI_APF


def adapt_rotation_matrix(
    R_MI_APF: np.ndarray, atom_pair: tuple[int, int], atomic_numbers, dipm_norm
) -> tuple:
    # Performs a rotation around the Y axis by 180 ° if nuclear charge of A
    # is smaller than B to achieve a consistent alignment
    # If A == B rotation depends on dipole moment
    A, B = atom_pair

    transpose = None

    if atomic_numbers[A] < atomic_numbers[B]:
        B, A = A, B

        transpose = [B, A]

        # rotation around the y axis
        R_MI_APF = -R_MI_APF
        R_MI_APF[1] = -R_MI_APF[1]

    elif atomic_numbers[A] == atomic_numbers[B]:
        if dipm_norm[A] < dipm_norm[B]:
            B, A = A, B

            transpose = [B, A]

            # rotation around the y axis
            R_MI_APF = -R_MI_APF
            R_MI_APF[1] = -R_MI_APF[1]

        elif dipm_norm[A] == dipm_norm[B]:
            print("Nucelar Charge and Dipole moment are the same.")

    atom_pair = A, B
    return R_MI_APF, atom_pair, transpose


def compute_rab(xyz, R_MI_APF, atom_pair, distance_mat):
    A, B = atom_pair
    xyz_A = xyz[A, :].copy()
    xyz_B = xyz[B, :].copy()

    s = 0.5 * (xyz_A + xyz_B)

    xyz_A -= s
    xyz_B -= s
    xyz_A = np.matmul(R_MI_APF, xyz_A)
    xyz_B = np.matmul(R_MI_APF, xyz_B)
    r_AB = (xyz_A - xyz_B).reshape(1, -1)
    R_AB = distance_mat[A, B]
    return r_AB, R_AB


# ______________________________________________________________________________
# Batched versions: all functions below take index arrays A, B of shape (P,) and
# process all P atom pairs at once. They reproduce `adapt_rotation_matrix` and
# `new_gen_pair_features` exactly (same values, same feature column order).
# ______________________________________________________________________________


def adapt_rotation_matrix_batched(
    R: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    atomic_numbers: np.ndarray,
    dipm_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batched `adapt_rotation_matrix`.

    Swaps A and B if Z_A < Z_B, or if Z_A == Z_B and |mu_A| < |mu_B|, and rotates
    the frame of swapped pairs by 180 deg around y.

    Args:
        R (np.ndarray): rotation matrices, (P, 3, 3).
        A, B (np.ndarray): atom indices, (P,).
        atomic_numbers (np.ndarray): (nat,).
        dipm_norm (np.ndarray): norms of the atomic dipole moments, (nat,).

    Returns:
        tuple: adapted R (P, 3, 3), new A (P,), new B (P,), swap mask (P,) bool.
    """
    zA, zB = atomic_numbers[A], atomic_numbers[B]
    swap = (zA < zB) | ((zA == zB) & (dipm_norm[A] < dipm_norm[B]))

    n_same = int(np.sum((zA == zB) & (dipm_norm[A] == dipm_norm[B])))
    if n_same:
        print(f"Nuclear charge and dipole moment are the same for {n_same} atom pair(s).")

    A_new = np.where(swap, B, A)
    B_new = np.where(swap, A, B)

    R = R.copy()
    R[swap] *= np.array([-1.0, 1.0, -1.0])[:, None]  # = -R, then row 1 negated back
    return R, A_new, B_new, swap


def _rotate_matrices_batched(R: np.ndarray, M: np.ndarray) -> np.ndarray:
    """R M R^T for every pair. R: (P, 3, 3), M: (P, 3, 3) or (P, n, 3, 3)."""
    if M.ndim == 4:
        R = R[:, None]  # broadcast over the n matrices of each pair
    return R @ M @ R.swapaxes(-1, -2)


def _atom_quantities_batched(
    R: np.ndarray,
    X: np.ndarray,
    scalars: np.ndarray,
    vectors: np.ndarray,
    matrices: np.ndarray,
    etot_idx: np.ndarray,
) -> np.ndarray:
    """Rotated atom-specific quantities of atom X for every pair (= `Quantity_AB[j]`).

    Returns:
        np.ndarray: (P, n_atom_features) with columns
            [matrices (lower triangle) | vectors | scalars | scalars / E_tot]
    """
    P = len(X)

    quad = _rotate_matrices_batched(R, matrices[X])  # (P, 4, 3, 3)
    quad = quad[:, :, TRIL[0], TRIL[1]].reshape(P, -1)  # (P, 4 * 6)

    dip = (vectors[X] @ R.swapaxes(-1, -2)).reshape(P, -1)  # (P, n_vec * 3)

    sc = scalars[X]  # (P, n_scalar)
    sc_rel = sc / sc[:, etot_idx[0], None]

    return np.concatenate([quad, dip, sc, sc_rel], axis=1)


def gen_pair_features_batched(
    A: np.ndarray, B: np.ndarray, arrays: tuple
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batched `new_gen_pair_features` for P atom pairs.

    Args:
        A, B (np.ndarray): atom indices of the pairs, (P,).
        arrays (tuple): molecular arrays (atomic_numbers, dipm_norm, xyz,
            distance_mat, q, dipm, qm, scalars, vectors, matrices, etot_idx,
            C6_params, wbo), see `transform_mlh.collect_feature_arrays`.

    Returns:
        tuple: features (P, n_feat), rotation matrices (P, 3, 3), A and B after the
            canonical reordering (P,), swap mask (P,) (True where A and B were swapped).
    """
    (
        atomic_numbers,
        dipm_norm,
        xyz,
        distance_mat,
        q,
        at_dipole,
        qm,
        scalars,
        vectors,
        matrices,
        etot_idx,
        C6_params,
        wbo,
    ) = arrays

    A = np.asarray(A)
    B = np.asarray(B)

    # ____Rotation into the atom pair frame (computed with the original order)____
    sup = supporting_vector_batched(at_dipole, A, B)
    R = get_atom_pair_rot_mat_batched(xyz, sup, A, B)
    R, A, B, swap = adapt_rotation_matrix_batched(R, A, B, atomic_numbers, dipm_norm)

    # ____compute_rab: R (xyz_A - s) - R (xyz_B - s) = R (xyz_A - xyz_B)____
    r_AB = np.einsum("pij,pj->pi", R, xyz[A] - xyz[B])  # (P, 3)
    R_AB = distance_mat[A, B]  # (P,)

    # ____Atom specific information____
    Q_A = _atom_quantities_batched(R, A, scalars, vectors, matrices, etot_idx)
    Q_B = _atom_quantities_batched(R, B, scalars, vectors, matrices, etot_idx)

    # ____Atom pair information____
    r_BA = -r_AB

    dipm_A = np.einsum("pij,pj->pi", R, at_dipole[A])  # (P, 3)
    dipm_B = np.einsum("pij,pj->pi", R, at_dipole[B])

    q_A = q[A]
    q_B = q[B]

    order1_aes = q_A * np.einsum("pi,pi->p", dipm_B, r_BA)
    order1_aes += q_B * np.einsum("pi,pi->p", dipm_A, r_AB)
    order1_aes /= R_AB**3

    qm_A = _rotate_matrices_batched(R, qm[A])  # (P, 3, 3)
    qm_B = _rotate_matrices_batched(R, qm[B])

    dA_r = np.einsum("pi,pi->p", dipm_A, r_AB)
    dB_r = np.einsum("pi,pi->p", dipm_B, r_AB)

    order2_aes = q_A * np.einsum("pi,pij,pj->p", r_AB, qm_B, r_AB)
    order2_aes += q_B * np.einsum("pi,pij,pj->p", r_AB, qm_A, r_AB)
    order2_aes -= 3 * dA_r * dB_r
    order2_aes += R_AB**2 * np.einsum("pi,pi->p", dipm_A, dipm_B)
    order2_aes /= R_AB**5

    C6_A = C6_params[A].astype(np.float64)  # stored as float32; float() in the per-pair version
    C6_B = C6_params[B].astype(np.float64)

    potE = q_A * q_B / R_AB

    pair_terms = np.stack(
        [potE, C6_A, C6_B, wbo[A, B], order1_aes, order2_aes], axis=1
    )  # (P, 6)

    features = np.concatenate(
        [
            Q_A,
            Q_B,
            pair_terms,
            Q_A * Q_B,  # Feature_Prod
            Q_A - Q_B,  # Feature_AbsDiff
            R_AB[:, None] ** R_AB_POWERS,  # (P, 7)
        ],
        axis=1,
    )

    return features, R, A, B, swap
