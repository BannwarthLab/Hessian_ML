import time

import numpy as np

from hess_ml.src.decorator.decorator import checkTiming
from hess_ml.src.geometry import Geometry
from hess_ml.src.rotation_func import Rotation_Functions


class FeatureGen(Geometry):
    @checkTiming(enabled=False)
    def gen_Feature(self, R_MI_APF, atom_A: int, atom_B: int):
        Features_temp = []

        A = atom_A
        B = atom_B

        self.check_list.append([A, B])

        # Performs a rotation around the X axis by 180 ° if nuclear charge of A is smaller than B to achieve a consistent alignment
        # If A == B rotation depends on dipole moment

        if self.NuclearCharge[A] < self.NuclearCharge[B]:
            B, A = A, B

            self.transpose_list.append([B, A])

            rot_Mat_X = self.rot_X(np.pi)

            R_MI_APF = np.matmul(rot_Mat_X, R_MI_APF)

            rot_Mat_Z = self.rot_Z(np.pi)

            R_MI_APF = np.matmul(rot_Mat_Z, R_MI_APF)

        elif self.NuclearCharge[A] == self.NuclearCharge[B]:
            if np.linalg.norm(self.dipm["A"][A]) < np.linalg.norm(self.dipm["A"][B]):
                B, A = A, B

                self.transpose_list.append([B, A])

                rot_Mat = self.rot_X(np.pi)

                R_MI_APF = np.matmul(rot_Mat, R_MI_APF)

                rot_Mat = self.rot_Z(np.pi)

                R_MI_APF = np.matmul(rot_Mat, R_MI_APF)

            elif np.linalg.norm(self.dipm["A"][A]) == np.linalg.norm(self.dipm["A"][B]):
                print("Nucelar Charge and Dipole moment are the same.")



        R_AB = np.linalg.norm(self.xyz[A, :] - self.xyz[B, :])

        Quantity_AB = [[], []]

        for j, atom in enumerate([A, B]):
            # ____Rotation from initial coordinate system to atom pair focused system____

            grad = np.matmul(R_MI_APF, self.gradient[atom])

            for qm_key in self.qm:
                temp_qm = np.zeros([3, 3])
                temp_qm[np.tril_indices(temp_qm.shape[0], k=0)] = self.qm[qm_key][atom]
                temp_qm = temp_qm + temp_qm.T - np.diag(np.diag(temp_qm))
                temp_qm = np.matmul(
                    np.matmul(R_MI_APF, temp_qm),
                    np.transpose(R_MI_APF),
                )
                Quantity_AB[j].extend((temp_qm[np.triu_indices(3)]).tolist())

            for dipm_key in self.dipm:
                Quantity_AB[j].extend(
                    np.matmul(R_MI_APF, self.dipm[dipm_key][atom]).tolist(),
                )

            for p_key in self.p:
                Quantity_AB[j].extend(self.p[p_key][atom].tolist())

            # ____Append Features to Feature Vector____

            Quantity_AB[j].extend(grad.tolist())
            Quantity_AB[j].extend(self.energy_based[atom].tolist())

            Quantity_AB[j].extend([self.NuclearCharge[atom]])

            Quantity_AB[j].extend([self.cn["default"][atom]])
            Quantity_AB[j].extend([self.cn["delta"][atom][0]])

            Quantity_AB[j].extend([self.q["default"][atom]])
            Quantity_AB[j].extend([self.q["delta"][atom][0]])

        Quantity_AB_arr = np.array(Quantity_AB)

        Feature_Arith = ((Quantity_AB_arr[0] + Quantity_AB_arr[1]) / 2).tolist()
        Feature_Prod = (Quantity_AB_arr[0] * Quantity_AB_arr[1]).tolist()
        Feature_AbsDiff = (Quantity_AB_arr[0] - Quantity_AB_arr[1]).tolist()

        Features_temp.extend(Quantity_AB[0])
        Features_temp.extend(Quantity_AB[1])

        Features_temp.extend(Feature_Arith)
        Features_temp.extend(Feature_Prod)
        Features_temp.extend(Feature_AbsDiff)
        Features_temp.extend([R_AB**12])
        Features_temp.extend([R_AB**6])
        Features_temp.extend([R_AB])
        Features_temp.extend([1 / R_AB])
        Features_temp.extend([1 / R_AB**6])

        return np.array(Features_temp)


class PredictProcess:
    def __init__(self) -> None:
        pass


class TransformPredict(Rotation_Functions, FeatureGen):
    @checkTiming(enabled=True)
    def Transform(self):
        self.R_MI_APF_mat = np.zeros([self.N_atoms * 3, self.N_atoms * 3])
        if self.do_calc:
            self.Feature_AB = []
            self.check_list = []
            self.transpose_list = []

            for atom_A in range(self.N_atoms):
                for atom_B in range(atom_A + 1, self.N_atoms):
                    xyz_temp = self.xyz.copy()

                    i0 = 3 * atom_A
                    i3 = 3 * atom_A + 3
                    j0 = 3 * atom_B
                    j3 = 3 * atom_B + 3

                    R_MI_APF = self.get_R_euler(
                        xyz_temp,
                        self.dipm["A"],
                        atom_A,
                        atom_B,
                    )

                    self.R_MI_APF_mat[i0:i3, j0:j3] = R_MI_APF

                    self.Feature_AB.append(self.gen_Feature(R_MI_APF, atom_A, atom_B))


class TransformTrain(Rotation_Functions, FeatureGen):
    @checkTiming(enabled=True)
    def Transform(self):
        self.Feature_AB = []
        self.Target_AB = []
        self.check_list = []
        self.transpose_list = []

        if self.do_calc:
            for atom_A in range(self.N_atoms):
                for atom_B in range(atom_A + 1, self.N_atoms):
                    xyz_temp = self.xyz.copy()
                    R_MI_APF = self.get_R_euler(
                        xyz_temp,
                        self.dipm["A"],
                        atom_A,
                        atom_B,
                    )

                    self.Feature_AB.append(self.gen_Feature(R_MI_APF, atom_A, atom_B))

                    i0 = 3 * atom_A
                    i3 = 3 * atom_A + 3
                    j0 = 3 * atom_B
                    j3 = 3 * atom_B + 3

                    H_APF = np.matmul(
                        np.matmul(R_MI_APF, self.target[i0:i3, j0:j3]),
                        np.transpose(R_MI_APF),
                    )  # Change Hessian

                    if [atom_A, atom_B] in self.transpose_list:
                        H_APF = np.matmul(
                            np.matmul(self.rot_X(np.pi), np.transpose(H_APF)),
                            np.transpose(self.rot_X(np.pi)),
                        )
                        H_APF = np.matmul(
                            np.matmul(self.rot_Z(np.pi), (H_APF)),
                            np.transpose(self.rot_Z(np.pi)),
                        )

                    self.Target_AB.append(list(H_APF.flatten()))
