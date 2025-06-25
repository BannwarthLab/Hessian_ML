from __future__ import annotations

import numpy as np
from copy import deepcopy 
from multiprocessing import Pool

from hess_ml.src2.utilities.decorator import checkTiming
from hess_ml.src2.utilities.geometrical import get_atom_pair_rot_mat
from hess_ml.src2.molecule.tblite.generation import FeatureCalculation
from hess_ml.src2.governance.globals import NUM_THREADS
import time as time

class Feature(FeatureCalculation):

    checkTiming(enabled=True)
    def get_processed_features(self):
        self.ImportFeature()
        self.Transform()

    def Transform(self):
        """Processes the features to ML readable.
        """
        if self._mol.nat == 1:
            
            self._mol.calc_succeeded = False
            
        if self._mol.calc_succeeded:
            self._processed_features = []
            self.R_MI_APF_mat = np.zeros([self._mol.nat * 3, self._mol.nat * 3])
            self.transpose_list = []

            #atom_pairs_comparison = np.array([(atom_A,atom_B) for atom_A in range(self._mol.nat) for atom_B in range(atom_A+1,self._mol.nat)])

            #print(atom_pairs_comparison.shape)

            temp_dist_mat = self._mol.feature.distance_mat.copy()
            temp_dist_mat[np.tril_indices_from(temp_dist_mat)] = np.inf
            atom_pairs = np.argwhere(temp_dist_mat<20)

            self._mol.computed_atom_pairs = atom_pairs

            pool = Pool(processes=NUM_THREADS)
            results_iterator = pool.map(self._transform_block, atom_pairs)
            pool.terminate()
            pool.join()

            R_MI_APFs,self._processed_features,transposes = zip(*deepcopy(results_iterator))

            self._processed_features = np.array(self._processed_features)

            transposes = list(transposes)

            for val in transposes:
                if val is not None:
                    self.transpose_list.append(val)

            for atom_pair,rot_mat in zip(atom_pairs,R_MI_APFs):

                atom_A,atom_B = atom_pair
                i0 = 3 * atom_A
                i3 = 3 * atom_A + 3
                j0 = 3 * atom_B
                j3 = 3 * atom_B + 3

                self.R_MI_APF_mat[i0:i3, j0:j3] = rot_mat


    def _transform_block(self,atom_pair:tuple)-> tuple:
        """Construction of feature vector for the prediction of the an AB Hessian matrix block.

        Args:
            index (int): index of for the list of atom pairs.
            atom_pairs (list[tuple]): list of atom pairs for the comp. of the rotation matrix.

        Returns:
            tuple: Rotation Matrix, Features, transpose info
        """        

        supporting_vector = self.supporting_vector(atom_pair)
        
        R_MI_APF = get_atom_pair_rot_mat(
            self._mol.xyz,
            supporting_vector,
            atom_pair
        )

        Feature_AB,transpose,R_MI_APF = self.gen_Feature(R_MI_APF, atom_pair)

        return R_MI_APF,Feature_AB,transpose

class ReducedFeature(Feature):
    def _transform_block(self,atom_pair:tuple)-> tuple:
        """Construction of feature vector for the prediction of the AB Hessian matrix block.

        Args:
            index (int): index of for the list of atom pairs
            atom_pairs (list[tuple]): list of atom pairs for the comp. of the rotation matrix

        Returns:
            tuple: Rotation Matrix, Features, transpose info
        """        

        supporting_vector = self.supporting_vector(atom_pair)

        R_MI_APF = get_atom_pair_rot_mat(
            self._mol.xyz,
            supporting_vector,
            atom_pair
        )

        Feature_AB,transpose,R_MI_APF = self.gen_Feature_red(R_MI_APF, atom_pair)

        return R_MI_APF,Feature_AB,transpose

class CustomFeature(Feature):
    def _transform_block(self,atom_pair:tuple)-> tuple:
        """Construction of feature vector for the prediction of the AB Hessian matrix block.

        Args:
            index (int): index of for the list of atom pairs
            atom_pairs (list[tuple]): list of atom pairs for the comp. of the rotation matrix

        Returns:
            tuple: Rotation Matrix, Features, transpose info
        """        
        

        supporting_vector = self.supporting_vector(atom_pair)
        
        R_MI_APF = get_atom_pair_rot_mat(
            self._mol.xyz,
            supporting_vector,
            atom_pair
        )

        Feature_AB,transpose,R_MI_APF = self.gen_Feature_custom(R_MI_APF, atom_pair)

        return R_MI_APF,Feature_AB,transpose