import numpy as np
import sys
from hess_ml.src.Rotation_func import Rotation_Functions
from hess_ml.src.Geometry import Geometry
from hess_ml.src.Features.Features import ImportFeatureTBlite


class FeatureGen(Geometry):

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

            rot_Mat = self.rot_X(np.pi)

            R_MI_APF = np.matmul(rot_Mat, R_MI_APF)

            rot_Mat = self.rot_Z(np.pi)

            R_MI_APF = np.matmul(rot_Mat, R_MI_APF)

        elif self.NuclearCharge[A] == self.NuclearCharge[B]:
            if np.linalg.norm(self.dipm["A"][A]) < np.linalg.norm(self.dipm["A"][B]):
                B, A = A, B

                self.transpose_list.append([B, A])

                rot_Mat = self.rot_X(np.pi)

                R_MI_APF = np.matmul(rot_Mat, R_MI_APF)

                rot_Mat = self.rot_Z(np.pi)

                R_MI_APF = np.matmul(rot_Mat, R_MI_APF)

        R_AB = np.linalg.norm(self.xyz[A, :] - self.xyz[B, :])

        Quantity_AB = [[], []]

        j = 0

        for atom in [A, B]:
            # ____Rotation from initial coordinate system to atom pair focused system____

            grad = np.matmul(R_MI_APF, self.gradient[atom])

            for qm_key in self.qm.keys():
                temp_qm = np.zeros([3, 3])
                temp_qm[np.tril_indices(temp_qm.shape[0], k=0)] = self.qm[qm_key][atom]
                temp_qm = temp_qm + temp_qm.T - np.diag(np.diag(temp_qm))
                temp_qm = np.matmul(
                    np.matmul(R_MI_APF, temp_qm), np.transpose(R_MI_APF)
                )

                Quantity_AB[j].extend((temp_qm[np.triu_indices(3)]))

            for dipm_key in self.dipm.keys():
                temp_dipm = self.dipm[dipm_key][atom]
                temp_dipm = np.matmul(R_MI_APF, temp_dipm)
                Quantity_AB[j].extend(temp_dipm)

            for q_key in self.q.keys():
                Quantity_AB[j].extend(self.q[q_key][atom])

            # ____Append Features to Feature Vector____

            Quantity_AB[j].extend(grad)
            Quantity_AB[j].extend(self.energy_based[atom])

            Quantity_AB[j].extend([self.NuclearCharge[atom]])

            Quantity_AB[j].extend([self.cn["default"][atom]])
            Quantity_AB[j].extend([self.cn["delta"][atom]])

            Quantity_AB[j].extend([self.p["default"][atom]])
            Quantity_AB[j].extend([self.p["delta"][atom]])

            j += 1

        Quantity_AB_arr = np.array(Quantity_AB)

        Feature_Arith = list((Quantity_AB_arr[0] + Quantity_AB_arr[1]) / 2)
        Feature_Prod = list(Quantity_AB_arr[0] * Quantity_AB_arr[1])
        Feature_AbsDiff = list((Quantity_AB_arr[0] - Quantity_AB_arr[1]))

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

        del Quantity_AB_arr
        del Quantity_AB

        del Feature_Arith
        del Feature_Prod
        del Feature_AbsDiff

        return np.array(Features_temp)


class PredictProcess(FeatureGen, ImportFeatureTBlite, Geometry, Rotation_Functions):
    def __init__(self):
        super().__init__()
        pass

    def transformFeature(self):

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
                        xyz_temp, self.dipm["A"], atom_A, atom_B
                    )
                    self.R_MI_APF_mat[i0:i3, j0:j3] = R_MI_APF
                    
                    self.Feature_AB.append(self.gen_Feature(R_MI_APF, atom_A, atom_B))

        return


class TrainProcess(FeatureGen, ImportFeatureTBlite, Geometry, Rotation_Functions):
    def __init__(self):
        super().__init__()
        pass

    def transformFeatureTarget(self):
        self.Feature_AB = []
        self.Target_AB = []
        self.check_list = []
        self.transpose_list = []
        if self.do_calc:
            for atom_A in range(self.N_atoms):
                for atom_B in range(atom_A + 1, self.N_atoms):
                    xyz_temp = self.xyz.copy()

                    R_MI_APF = self.get_R_euler(
                        xyz_temp, self.dipm["A"], atom_A, atom_B
                    )

                    self.Feature_AB.append(self.gen_Feature(R_MI_APF, atom_A, atom_B))

                    i0 = 3 * atom_A
                    i3 = 3 * atom_A + 3
                    j0 = 3 * atom_B
                    j3 = 3 * atom_B + 3

                    H_APF = np.matmul(
                        np.matmul(R_MI_APF, self.target[i0:i3, j0:j3].copy()),
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

        return
