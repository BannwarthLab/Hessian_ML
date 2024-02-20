from __future__ import annotations

import numpy as np
from numpy import linalg
from scipy.spatial.distance import pdist

import hess_ml.src.constants.constants as const
from hess_ml.src.decorator.decorator import checkTiming


class Rotation_Functions:
    def __init__(self):
        pass

    def angle_two_vec(self, a, b):
        cosangle = 0 if linalg.norm(a) == 0 or linalg.norm(b) == 0 else np.dot(a, b) / linalg.norm(a) / linalg.norm(b)
        return np.arccos(np.clip(cosangle, -1, 1))


    def center_charge(self, coord_var):
        d = np.zeros(3)
        charge_sum = 0
        for i in range(len(coord_var["atoms"])):
            charge = const.ELEMENTS2Z[coord_var.loc[i, "atoms"]]
            d += charge * coord_var.iloc[i, 1:]
            charge_sum += charge
        return d / charge_sum

    def center_mass(self, coord_var):
        d = np.zeros(3)
        mass_sum = 0
        for i in range(len(coord_var["atoms"])):
            mass = const.elements_dict[coord_var.loc[i, "atoms"]]
            d += mass * coord_var.iloc[i, 1:]
            mass_sum += mass
        return d / mass_sum

    def check_eig_vec(self, eig_vec):
        if linalg.det(eig_vec) < 0:
            for i in range(3):
                eig_vec[2, i] = -eig_vec[2, i]
        return eig_vec

    def coord_rot(self, coord_var, rotM):
        for i in range(len(coord_var.iloc[:, 1])):
            coord_var.iloc[i, 1:] = np.matmul(rotM, coord_var.iloc[i, 1:])
        return coord_var

    def rot_gradient(self, R):
        for i in range(self.N_atoms):
            self.gradient[i] = np.matmul(R, self.gradient[i])

    def vec_trans(self, coord_var, trans):
        coord_var_new = coord_var.copy()
        coord_var_new.iloc[:, 1:] = np.array(coord_var.iloc[:, 1:]) - np.array(trans)
        return coord_var_new

    def get_R_kabsch(self, xyz, dipm, i, j):
        T = 0.5 * (xyz[i, :] + xyz[j, :])

        dipm[i, :] + dipm[j, :]

        xyz -= T

        dist_r = np.dot(xyz[i, :], xyz[j, :])

        align_vec = np.zeros([2, 3])
        align_vec[:, 2] = np.array([dist_r * 0.5, dist_r * 0.5])

    def _check_x_axis_error(self,vec_x,i,j,coord_th = 1e-8):

        if np.abs(vec_x[1]) > coord_th or np.abs(vec_x[2]) > coord_th:
            print(vec_x)
            print(f"Error in vec_x for {i,j}")
    def _check_xyz_error(self,xyz,i,j,rotation_matrix,coord_th=1e-8):
        xyz[i, :] = np.matmul(rotation_matrix, xyz[i, :])
        xyz[j, :] = np.matmul(rotation_matrix, xyz[j, :])

        if xyz[i, 0] > coord_th or xyz[i, 1] > coord_th:
            print(f"Error in coord i:{i,j}")
            print(xyz[[i, j], :])

        if xyz[j, 0] > coord_th or xyz[j, 1] > coord_th:
            print(f"Error in coord j:{i,j}")
            print(xyz[[i, j], :])
    # ____Uses for i=j the mean of the xyz's atoms as an artifical atom____
    @checkTiming(enabled=False)
    def get_R_euler(self, coord_end, dipm, i, j):
        zero_th = 0.0
        sum_th = 1e-12
        if i == j:
            print("error")

        axis = np.identity(3)

        T = 1 / 2 * coord_end[i, :] + 1 / 2 * coord_end[j, :]

        vec_dipm = (
            dipm[i, :] + dipm[j, :]
        )  # np.sum(dipm.iloc[:,1:])/len(dipm.iloc[:,1:])#

        coord_end -= T

        # Rotation for i < j

        # Atom pair focussed coordinate system
        vec_z = coord_end[i, :]
        vec_x = np.cross(vec_z, vec_dipm)

        LL = np.cross(vec_z, axis[2])

        if np.sum(np.abs(np.array(coord_end[[i, j], :2]))) < sum_th:
            alpha = 2 * np.pi - self.angle_two_vec(vec_x, axis[0])
            beta = self.angle_two_vec(vec_z, axis[2])
            gamma = 0

            if linalg.det(np.array([axis[0], axis[2], vec_x])) > zero_th:
                alpha = 2 * np.pi - alpha

        else:
            alpha = self.angle_two_vec(LL, axis[0])
            beta = self.angle_two_vec(vec_z, axis[2])
            gamma = self.angle_two_vec(LL, vec_x)

            # Find right rotation angle
            if linalg.det(np.array([axis[0], axis[2], LL])) < zero_th:
                alpha = 2 * np.pi - alpha

            if linalg.det(np.array([LL, vec_x, vec_z])) > zero_th:
                gamma = 2 * np.pi - gamma

        R_euler = np.matmul(
            np.matmul(self.rot_Z(gamma), self.rot_X(beta)),
            self.rot_Z(alpha),
        )

        # Rotation by 180 ° if dipole moment is negative in x

        if np.matmul(R_euler, vec_x)[0] < zero_th:
            R_z = self.rot_Z(np.pi)
            R_euler = np.matmul(R_z, R_euler)

        # Apply the euler rotation matrix on the new x axis for verification reasons
        vec_x = np.matmul(R_euler, vec_x)

        self.angle_two_vec(LL, vec_x)

        # Check for Errors in dipole moment or the coordinates

        if vec_x[0] < zero_th:
            print(f"Error in vec_x[0] in {i,j}")

        self._check_x_axis_error(vec_x,i,j)

        self._check_xyz_error(coord_end,i,j,R_euler,coord_th=1e-8)

        return R_euler


    def rot_Z(self, alpha):  # Givens rotation around the z-axis
        return np.array(
            [
                [np.cos(alpha), -np.sin(alpha), 0],
                [np.sin(alpha), np.cos(alpha), 0],
                [0, 0, 1],
            ],
        )

    def rot_X(self, alpha):  # Givens rotation around the x-axis
        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(alpha), -np.sin(alpha)],
                [0, np.sin(alpha), np.cos(alpha)],
            ],
        )

    def rot_Y(self, alpha):  # Givens rotation around the y-axis
        return np.array(
            [
                [np.cos(alpha), 0.0, -np.sin(alpha)],
                [0.0, 1, 0.0],
                [np.sin(alpha), 0.0, np.cos(alpha)],
            ],
        )

    def eig_vec_rot(
        self,
        eig_vec,
    ):  # Checks for the highest value of the eigenvector matrix exchanges if the highest is not in first place
        for i in [0, 1]:
            max_abs_val = max(eig_vec[i].min(), eig_vec[i].max(), key=abs)
            if max_abs_val < 0:
                eig_vec[i] = -eig_vec[i]
                eig_vec[i + 1] = -eig_vec[i + 1]

        return eig_vec

    def inert_tensor(self, coord_var):  # computes the inert tensor
        inert_t = np.zeros([3, 3])

        m = 0
        for i in range(len(coord_var.iloc[:, 1])):
            mi = const.elements_dict[coord_var.iloc[i, 0]]
            xi = coord_var.iloc[i, 1]
            yi = coord_var.iloc[i, 2]
            zi = coord_var.iloc[i, 3]
            m += mi

            txx = mi * (yi**2 + zi**2)
            txy = -mi * xi * yi
            txz = -mi * xi * zi
            tyy = mi * (xi**2 + zi**2)
            txz = -mi * xi * zi
            tyz = -mi * yi * zi
            tzz = mi * (yi**2 + xi**2)

            inert_t += np.array([[txx, txy, txz], [txy, tyy, tyz], [txz, tyz, tzz]])

        return inert_t / m / const.bohr2angs**2

    def rotM_hess(self, R, coord_var):
        P = np.zeros([3 * len(coord_var["atoms"]), 3 * len(coord_var["atoms"])])
        for i in range(len(coord_var["atoms"])):
            P[3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)] = R
        return P

    def rotM_hess2(self, R, n):
        P = np.zeros([3 * n, 3 * n])
        for i in range(n):
            P[3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)] = R
        return P

    def vector_rot(coord_var, rotM):
        coord_var_new = coord_var.copy()
        for i in range(len(coord_var.iloc[:, 1])):
            coord_var_new.iloc[i, :] = np.matmul(rotM, coord_var.iloc[i, :])
        return coord_var_new

    def calc_R(self, coord):
        ############
        ########### Rotation of coordinates and hessian into intermediate position
        # Calculating center of mass
        s = self.center_mass(coord)

        # Translation of coordinate system int center of mass
        coord = self.vec_trans(coord, s)
        # vec_trans(dipm,s)

        # Calculating moment of inertia
        inert_tens = self.inert_tensor(coord)
        # Calculating eigenvalues and eigenvectors
        eig_val, eig_vec = linalg.eigh(inert_tens)

        # Check if the coordinate system is right-handed --> important for chirality
        eig_vec = self.check_eig_vec(eig_vec)

        # Rotating eigenvectors, so that highest values are positive

        eig_vec = self.eig_vec_rot(eig_vec)

        return eig_vec, coord

    def qm_matrix(qm_atom):
        qm_matrix_list = []
        for i in range(len(qm_atom)):
            xx = qm_atom[i, 0]
            yy = qm_atom[i, 1]
            zz = qm_atom[i, 2]
            xy = qm_atom[i, 3]
            xz = qm_atom[i, 4]
            yz = qm_atom[i, 5]

            qm_matrix = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])

            qm_matrix_list.append(qm_matrix)

        return qm_matrix_list
