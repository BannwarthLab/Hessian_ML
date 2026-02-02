import numpy as np
from mlhess.utils.math.matrix_operation import rotate_matrix, rotate_vector_array

# if TYPE_CHECKING:
#     from mlhess.machinelearning.feature.base_class import Feature


def gen_pair_features(self, R_MI_APF, atom_pair: tuple[int, int]) -> tuple:
    """Generate Features for an atom pair.

    Args:
        R_MI_APF (_type_): Rotation matrix
        atom_pair (tuple[int,int]): index of atom A and B

    Returns:
        tuple: Features,transpose, rotation matrix
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
            quad_moments[:, np.tril_indices(3)[0], np.tril_indices(3)[1]]
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

    Features_temp.extend(R_AB ** np.array([-12, 6, 1, -1, -3, -6, -12]))

    return np.array(Features_temp), transpose, R_MI_APF
