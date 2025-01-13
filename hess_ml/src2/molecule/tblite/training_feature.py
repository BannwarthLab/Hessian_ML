from __future__ import annotations

from multiprocessing import Pool
import numpy as np
import os 

from hess_ml.src.decorator.decorator import checkTiming
from hess_ml.src2.molecule.molecule import Molecule
from hess_ml.src2.molecule.tblite.generation import FeatureCalculation
from hess_ml.src2.utilities.geometrical import get_atom_pair_rot_mat
from hess_ml.src2.utilities.matrix_operation import rotate_matrix

from hess_ml.src2.governance.globals import NUM_THREADS

class Feature(FeatureCalculation):
    """A class for the transformation of Hessian matrix and its features for training ML models."""
    def get_processed_features(self):
        self.ImportFeature()
        self.Transform()

    @checkTiming(enabled=False)
    def Transform(self):
        """Transforms the Hessian matrix and features of a molecule into a target-descriptor relation for training."""

        self.processed_target = []
        self._processed_features = []
        if self._mol.nat == 1:
            self._mol.calc_succeeded = False

        if self._mol.calc_succeeded:
            self.transpose_list = []

#            atom_pairs = [(atom_A,atom_B) for atom_A in range(self._mol.nat) for atom_B in range(atom_A+1,self._mol.nat)]

            temp_dist_mat = self._mol.feature.distance_mat.copy()
            temp_dist_mat[np.tril_indices_from(temp_dist_mat)] = np.inf
            atom_pairs = np.argwhere(temp_dist_mat<20)

            self._mol.computed_atom_pairs = atom_pairs

            pool = Pool(processes=NUM_THREADS)
            results_iterator = pool.map(self._transform_block, atom_pairs)

            transposes = self.process_results(results_iterator)

            pool.terminate()
            pool.join()

            for val in transposes:
                if val is not None:
                    self.transpose_list.append(val)

            del atom_pairs, pool, results_iterator
        self._processed_features = np.array(self._processed_features)

    def process_results(self,results)-> list:
        # Process and release memory for results incrementally
        self.processed_target.extend(result[0] for result in results)
        self._processed_features.extend(result[1] for result in results)
        transposes = [result[2] for result in results]
        # Release memory for results
        del results

        return transposes

    def _transform_block(self,atom_pair:list):
        """Rotation of an AB Hessian matrix block and the construction its feature vector.

        :param index: index of for the list of atom pairs
        :param atom_pairs: list of atom pairs for the comp. of the rotation matrix"""

        atom_A,atom_B = atom_pair
        supporting_vector = self.supporting_vector(atom_pair)

        R_MI_APF = get_atom_pair_rot_mat(
            self._mol.xyz,
            supporting_vector,
            atom_pair,
        )

        Feature_AB,transpose,R_MI_APF = self.gen_Feature(R_MI_APF,atom_pair)

        i0 = 3 * atom_A
        i3 = 3 * atom_A + 3
        j0 = 3 * atom_B
        j3 = 3 * atom_B + 3

        H_APF = rotate_matrix(R_MI_APF,self._mol.hessian.hessian[i0:i3, j0:j3])

        if transpose is not None:
            H_APF = H_APF.T

        return list(H_APF.flatten()),Feature_AB,transpose


class ReducedFeature(Feature):
    def _transform_block(self,atom_pair:list):
        """Rotation of an AB Hessian matrix block and the construction its feature vector.

        :param index: index of for the list of atom pairs
        :param atom_pairs: list of atom pairs for the comp. of the rotation matrix"""

        atom_A,atom_B = atom_pair
        supporting_vector = self.supporting_vector(atom_pair)

        R_MI_APF = get_atom_pair_rot_mat(
            self._mol.xyz,
            supporting_vector,
            atom_pair,
        )

        Feature_AB,transpose,R_MI_APF = self.gen_Feature_red(R_MI_APF,atom_pair)

        i0 = 3 * atom_A
        i3 = 3 * atom_A + 3
        j0 = 3 * atom_B
        j3 = 3 * atom_B + 3

        H_APF = rotate_matrix(R_MI_APF,self._mol.hessian.hessian[i0:i3, j0:j3])

        if transpose is not None:
            H_APF = H_APF.T

        return list(H_APF.flatten()),Feature_AB,transpose
    
    
class CustomFeature(Feature):
    def _transform_block(self,atom_pair:list):
        """Rotation of an AB Hessian matrix block and the construction its feature vector.

        :param index: index of for the list of atom pairs
        :param atom_pairs: list of atom pairs for the comp. of the rotation matrix"""

        atom_A,atom_B = atom_pair
        supporting_vector = self.supporting_vector(atom_pair)

        R_MI_APF = get_atom_pair_rot_mat(
            self._mol.xyz,
            supporting_vector,
            atom_pair,
        )

        Feature_AB,transpose,R_MI_APF = self.gen_Feature_custom(R_MI_APF,atom_pair)
        
        i0 = 3 * atom_A
        i3 = 3 * atom_A + 3
        j0 = 3 * atom_B
        j3 = 3 * atom_B + 3

        H_APF = rotate_matrix(R_MI_APF,self._mol.hessian.hessian[i0:i3, j0:j3])

        if transpose is not None:
            H_APF = H_APF.T

        return list(H_APF.flatten()),Feature_AB,transpose

