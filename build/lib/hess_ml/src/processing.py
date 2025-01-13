from __future__ import annotations

import gc
import sys
import time
from copy import deepcopy
from functools import partial
from multiprocessing import Array, Pool, Process
from typing import TYPE_CHECKING

import numpy as np

from hess_ml.src.decorator.decorator import checkTiming
from hess_ml.src.rotation_func import Rotation_Functions

if TYPE_CHECKING:
    from hess_ml.src.template import TestMLHessianGFN2xTB, TrainMLHessianGFN2xTB

class FeatureGen:
    @checkTiming(enabled=False)
    def gen_Feature(self:TrainMLHessianGFN2xTB, R_MI_APF, atom_A: int, atom_B: int) -> tuple:
        """Generate Features for an atom pair.

        Args:
            R_MI_APF (_type_): Rotation matrix
            atom_A (int): Index of atom A
            atom_B (int): Index of atom B

        Returns:
            tuple: Features,transpose, rotation matrix
        """
        Features_temp = []

        A = atom_A
        B = atom_B

        transpose = None

        # Performs a rotation around the Y axis by 180 ° if nuclear charge of A
        # is smaller than B to achieve a consistent alignment
        # If A == B rotation depends on dipole moment

        if self.NuclearCharge[A] < self.NuclearCharge[B]:
            B, A = A, B

            transpose = [B, A]

            R_y= -np.eye(3)
            R_y[1,1] = 1

            R_MI_APF = np.matmul(R_y, R_MI_APF)

        elif self.NuclearCharge[A] == self.NuclearCharge[B]:

            if np.linalg.norm(self.dipm["A"][A]) < np.linalg.norm(self.dipm["A"][B]):
                B, A = A, B

                transpose = [B, A]

                R_y =  -np.eye(3)
                R_y[1,1] = 1

                R_MI_APF = np.matmul(R_y, R_MI_APF)

            elif np.linalg.norm(self.dipm["A"][A]) == np.linalg.norm(self.dipm["A"][B]):
                print("Nucelar Charge and Dipole moment are the same.")

        r_AB = (self.xyz[A, :].copy() - self.xyz[B, :].copy()).reshape(1,-1)

        r_AB = np.matmul(R_MI_APF,r_AB.T).T

        R_AB = self.distance_mat[A,B]

        Quantity_AB = [[], []]

        #Atom specific information
        for j, atom in enumerate([A, B]):
            # ____Rotation from initial coordinate system to atom pair focused system____

            grad = np.matmul(R_MI_APF, self.gradient[atom])

            #This is i an 7 x 3 x 3 matrix, rotation of all matrices is achieved with this routine
            quad_moments = self.rotate_matrix(R_MI_APF,self.matrices[atom])
            #Only takes the lower triangle values as the matrices are symmetric
            Quantity_AB[j].extend(quad_moments[:,np.tril_indices(3)[0],np.tril_indices(3)[1]].flatten().tolist())

            dipole_moments = self.rotate_vector_array(R_MI_APF,np.array(self.vectors[atom]))
            Quantity_AB[j].extend(dipole_moments.flatten().tolist())

            # ____Append Features to Feature Vector____

            Quantity_AB[j].extend(grad.tolist())
            Quantity_AB[j].extend(self.scalars[atom])
            Quantity_AB[j].extend([self.NuclearCharge[atom]])

        Quantity_AB_arr = np.array(Quantity_AB)

        Feature_Arith = ((Quantity_AB_arr[0] + Quantity_AB_arr[1]) / 2).tolist()
        Feature_Prod = (Quantity_AB_arr[0] * Quantity_AB_arr[1]).tolist()
        Feature_AbsDiff = (Quantity_AB_arr[0] - Quantity_AB_arr[1]).tolist()

        #atom pair information

        r_BA = -r_AB

        dipm_key = "A"

        dipm_A = self.dipm[dipm_key][A].reshape(1,-1)
        dipm_B = self.dipm[dipm_key][B].reshape(1,-1)

        dipm_A = np.matmul(R_MI_APF,dipm_A.T).T
        dipm_B = np.matmul(R_MI_APF,dipm_B.T).T

        q_A = self.q["default"][A]
        q_B = self.q["default"][B]

        order1_aes = q_A*np.dot(dipm_B,r_BA.T) + q_B*np.dot(dipm_A,r_AB.T)
        order1_aes /= R_AB**3

        qm_key = "A"

        qm_A = self.qm[qm_key][A]
        qm_A = self.rotate_matrix(R_MI_APF,qm_A)

        qm_B = self.qm[qm_key][B]

        qm_B = self.rotate_matrix(R_MI_APF,qm_B)

        order2_aes = q_A*np.matmul(r_AB,np.matmul(qm_B,r_AB.T))
        order2_aes += q_B*np.matmul(r_AB,np.matmul(qm_A,r_AB.T))
        order2_aes -= 3*np.dot(dipm_A,r_AB.T)*np.dot(dipm_B,r_AB.T)
        order2_aes += R_AB**2*np.dot(dipm_A,dipm_B.T)

        order2_aes /= R_AB**5

        C6_A = float(self.C6_params[A])
        C6_B = float(self.C6_params[B])

        potE = q_A*q_B/R_AB

        atoms = [A,B]
        wbo_th = 0.25

        for atom in atoms:
            wbo_r = np.zeros(3)
            nuc_charge_loc = np.zeros(3)
            n_adj =  0

            for idx in range(self.N_atoms):
                if idx not in atoms and self.wbo[atom,idx] > wbo_th:
                    n_adj += 1
                    temp_r_ab = self.xyz[idx].copy()-self.xyz[atom].copy()
                    r_ab_norm = np.linalg.norm(temp_r_ab)
                    wbo_r += self.wbo[atom,idx]/r_ab_norm**3 * np.matmul(R_MI_APF,temp_r_ab)
                    nuc_charge_loc += self.NuclearCharge[idx]*self.NuclearCharge[atom]/r_ab_norm**3 * np.matmul(R_MI_APF,temp_r_ab)
                    #wbo_r_norm += np.linalg.norm(wbo_r)
                    #loc_nuc_charge_norm += np.linalg.norm(nuc_charge_loc)


            mask = np.ones(self.N_atoms,dtype=bool)

            mask[atom] = False

            V_J = np.sum(self.q["default"][mask]/self.distance_mat[atom,mask])

            Features_temp.append(V_J*self.q["default"][atom])

            Features_temp.append(n_adj)

            Features_temp.extend(wbo_r)
            #Features_temp.append(wbo_r_norm)

            Features_temp.extend(nuc_charge_loc)
            #Features_temp.append(loc_nuc_charge_norm)

        Features_temp.append(potE)

        Features_temp.append(C6_A)
        Features_temp.append(C6_B)

        Features_temp.append(self.wbo[A,B])

        Features_temp.extend(r_AB.tolist()[0])

        Features_temp.extend(order1_aes[0])
        Features_temp.extend(order2_aes[0])

        Features_temp.extend(Quantity_AB[0])
        Features_temp.extend(Quantity_AB[1])

        Features_temp.extend(Feature_Arith)
        Features_temp.extend(Feature_Prod)
        Features_temp.extend(Feature_AbsDiff)

        for i in [12,6,1,-1,-2,-3,-6]:
            Features_temp.extend([R_AB**i])

        return np.array(Features_temp),transpose,R_MI_APF

    def get_start_specific_key(self,keys,starting_string):

        for key in keys:
            if key.startswith(starting_string):
                print(key)
                break

        return key

class PredictProcess:
    def __init__(self) -> None:
        pass

class TransformPredict(Rotation_Functions, FeatureGen):
    @checkTiming(enabled=True)
    def Transform(self:TestMLHessianGFN2xTB):
        """Processes the features to ML readable.
        """
        if self.do_calc:
            num_cpus = self.general_config.threads
            self.Feature_AB = []
            self.R_MI_APF_mat = np.zeros([self.N_atoms * 3, self.N_atoms * 3])
            self.transpose_list = []

            atoms = [(atom_A,atom_B) for atom_A in range(self.N_atoms) for atom_B in range(atom_A+1,self.N_atoms)]

            partial_func = partial(self._transform_block,atom_pairs=atoms)

            indices = list(range(len(atoms)))

            with Pool(processes=num_cpus) as pool:
                results = pool.map(partial_func,indices)

            R_MI_APFs,self.Feature_AB,transposes = zip(*deepcopy(results))

            self.Feature_AB = list(self.Feature_AB)

            transposes = list(transposes)

            for val in transposes:
                if val is not None:
                    self.transpose_list.append(val)

            for atom_pair,rot_mat in zip(atoms,R_MI_APFs):

                atom_A,atom_B = atom_pair
                i0 = 3 * atom_A
                i3 = 3 * atom_A + 3
                j0 = 3 * atom_B
                j3 = 3 * atom_B + 3

                self.R_MI_APF_mat[i0:i3, j0:j3] = rot_mat

    def _transform_block(self:TestMLHessianGFN2xTB,index:int,atom_pairs:list[tuple])-> tuple:
        """Construction of feature vector for the prediction of the an AB Hessian matrix block.

        Args:
            index (int): index of for the list of atom pairs
            atom_pairs (list[tuple]): list of atom pairs for the comp. of the rotation matrix

        Returns:
            tuple: Rotation Matrix, Features, transpose info
        """

        atom_A,atom_B = atom_pairs[index]
        xyz = self.xyz.copy()

        R_MI_APF = self.get_R_euler(
            xyz,
            self.dipm["A"],
            atom_A,
            atom_B,
        )

        Feature_AB,transpose,R_MI_APF = self.gen_Feature(R_MI_APF, atom_A, atom_B)

        return R_MI_APF,Feature_AB,transpose


class TransformTrain(Rotation_Functions, FeatureGen):
    """A class for the transformation of Hessian matrix and its features for training ML models."""
    @checkTiming(enabled=True)
    def Transform(self:TrainMLHessianGFN2xTB):
        """Transforms the Hessian matrix and features of a molecule into a target-decriptor relation for training."""

        self.Feature_AB = []
        self.Target_AB = []

        if self.do_calc:
            num_cpus = self.general_config.threads
            self.transpose_list = []

            atoms = [(atom_A,atom_B) for atom_A in range(self.N_atoms) for atom_B in range(atom_A+1,self.N_atoms)]

            pool = Pool(processes=num_cpus)

            results_iterator = pool.map(self._transform_block, atoms)

            transposes = self.process_results(results_iterator)

            pool.terminate()
            pool.join()

            for val in transposes:
                if val is not None:
                    self.transpose_list.append(val)

            del atoms, pool, results_iterator,num_cpus


    def process_results(self,results):
        # Process and release memory for results incrementally
        self.Target_AB.extend(result[0] for result in results)
        self.Feature_AB.extend(result[1] for result in results)
        transposes = [result[2] for result in results]
        # Release memory for results
        del results

        return transposes

    def _transform_block(self:TrainMLHessianGFN2xTB,atom_pair:list):
        """Rotation of an AB Hessian matrix block and the construction its feature vector.

        :param index: index of for the list of atom pairs
        :param atom_pairs: list of atom pairs for the comp. of the rotation matrix"""

        atom_A,atom_B = atom_pair

        R_MI_APF = self.get_R_euler(
            self.xyz.copy(),
            self.dipm["A"],
            atom_A,
            atom_B,
        )

        Feature_AB,transpose,R_MI_APF = self.gen_Feature(R_MI_APF, atom_A, atom_B)

        i0 = 3 * atom_A
        i3 = 3 * atom_A + 3
        j0 = 3 * atom_B
        j3 = 3 * atom_B + 3

        H_APF = self.rotate_matrix(R_MI_APF,self.target[i0:i3, j0:j3])

        if transpose is not None:
            H_APF = H_APF.T

        return list(H_APF.flatten()),Feature_AB,transpose
