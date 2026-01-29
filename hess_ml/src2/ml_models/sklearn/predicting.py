from __future__ import annotations

import copy
import os
import sys

import time
from typing import TYPE_CHECKING

import torch
import numpy as np
from joblib import load
from tcgm_lib.IO.writer import wrt_hess_to_xtb

from hess_ml.src2.utilities.decorator import checkTiming
from hess_ml.src2.molecule.molecule import Molecule
from hess_ml.src2.molecule.tblite.prediction_feature import ReducedFeature
from hess_ml.src2.molecule.tblite.prediction_feature import CustomFeature

if TYPE_CHECKING:
    from sklearn.dummy import DummyRegressor
    from hess_ml.src2.governance.config import Configurations

class Predictor:
    def __init__(self,config:Configurations) -> None:
        self.config = config
        self._model: DummyRegressor | None = None

    @property
    def model(self) -> DummyRegressor:
        if self._model is not None:
            return self._model

        if os.path.isfile(f"{self.config.predict.model_name}.joblib"):
            self._model = load(f"{self.config.predict.model_name}.joblib")
            return self._model

        err_msg = "No ML model found."
        raise AttributeError(err_msg)
    
    @model.setter
    def model(self,value)->None:
        self._model = value

    @checkTiming(enabled=True)
    def predict(self,mol:Molecule):
        return mol
    
    def load_model(self,fname):
        self._model = load(fname)

class HessianPredictor(Predictor):

    @checkTiming(enabled=True)
    def predict(self, mol:Molecule):
        """Predict the Hessian of a molecule from its features.

        Args:
            mol (Molecule): Molecule
        """
        if mol.nat == 1:
            mol.calc_succeeded = False

        feat = mol.feature.processed_features.astype(np.float32)
        
        if mol.calc_succeeded:
            upper_tri_blocks_hessian = self.model.predict(feat)

            mol.ml_hessian.hessian = self.gen_hess_from_vec_pred_damped(
                upper_tri_blocks_hessian,
                mol.nat,
                mol.feature.R_MI_APF_mat,
                mol.feature.transpose_list,
                mol.computed_atom_pairs,
                mol.feature.distance_mat,
            )

            del upper_tri_blocks_hessian

        return mol

    def gen_hess_from_vec_pred(
                self, hess_vec_ab, N_atoms, R_MI_APF_mat, transpose_list,
            ):

            atom_indices = np.array([
                (atom_A, atom_B) for atom_A in range(N_atoms) for atom_B in range(atom_A + 1, N_atoms)
            ])

            # Convert transpose_list to a set for faster membership checking
            transpose_set = set(map(tuple, transpose_list))

            # Use vectorized comparison to check if each (atom_A, atom_B) pair is in transpose_set
            transposes = np.array([tuple(pair) in transpose_set for pair in atom_indices])

            rabs,hess_ab = restructure_RH(np.array(hess_vec_ab), N_atoms, R_MI_APF_mat, np.array(transposes))
            
            
            hess_ab = np.einsum('mij,mjk,mlk->mil', rabs, hess_ab, rabs)

            Hessian = np.zeros([N_atoms * 3, N_atoms * 3])

            ite_hetero = 0
            for atom_A in range(N_atoms):
                for atom_B in range(atom_A + 1, N_atoms):
                    Hessian[
                            3 * atom_A : 3 * atom_A + 3, 3 * atom_B : 3 * atom_B + 3,
                        ] = hess_ab[ite_hetero]

                    Hessian[
                            3 * atom_B : 3 * atom_B + 3, 3 * atom_A : 3 * atom_A + 3,
                        ] = Hessian[
                            3 * atom_A : 3 * atom_A + 3, 3 * atom_B : 3 * atom_B + 3,
                        ].T 

                    ite_hetero += 1 

            for atom_A in range(N_atoms):
                for atom_B in range(N_atoms):
                    if atom_A != atom_B:
                        Hessian[
                            3 * atom_A : 3 * atom_A + 3, 3 * atom_A : 3 * atom_A + 3,
                        ] -= Hessian[
                            3 * atom_A : 3 * atom_A + 3, 3 * atom_B : 3 * atom_B + 3,
                        ]

                Hessian[3 * atom_A : 3 * atom_A + 3, 3 * atom_A : 3 * atom_A + 3] = (
                    Hessian[3 * atom_A : 3 * atom_A + 3, 3 * atom_A : 3 * atom_A + 3]
                    + np.transpose(
                        Hessian[3 * atom_A : 3 * atom_A + 3, 3 * atom_A : 3 * atom_A + 3],
                    )
                ) / 2   


            return Hessian
    
    def damping(self,dist):
        return 1/(np.exp((dist-13.0)/0.1)+1)

    def gen_hess_from_vec_pred_damped(
                self, hess_vec_ab, N_atoms, R_MI_APF_mat, transpose_list, atom_pairs, dist_mat
            ):

            # atom_indices = np.array([
            #     (atom_A, atom_B) for atom_A in range(N_atoms) for atom_B in range(atom_A + 1, N_atoms)
            # ])

            # Convert transpose_list to a set for faster membership checking
            transpose_set = set(map(tuple, transpose_list))

            # Use vectorized comparison to check if each (atom_A, atom_B) pair is in transpose_set
            transposes = np.array([tuple(pair) in transpose_set for pair in atom_pairs])

            rabs,hess_ab = restructure_RH(np.array(hess_vec_ab), atom_pairs, R_MI_APF_mat, np.array(transposes))

            hess_ab = np.einsum('mij,mjk,mlk->mil', rabs, hess_ab, rabs)

            Hessian = np.zeros([N_atoms * 3, N_atoms * 3])

            ite_hetero = 0

            for atom_A,atom_B in zip(atom_pairs[:,0],atom_pairs[:,1]):
                Hessian[
                        3 * atom_A : 3 * atom_A + 3, 3 * atom_B : 3 * atom_B + 3,
                    ] = hess_ab[ite_hetero] * self.damping(dist_mat[atom_A,atom_B])

                Hessian[
                        3 * atom_B : 3 * atom_B + 3, 3 * atom_A : 3 * atom_A + 3,
                    ] = Hessian[
                        3 * atom_A : 3 * atom_A + 3, 3 * atom_B : 3 * atom_B + 3,
                    ].T 

                ite_hetero += 1 

            for atom_A in range(N_atoms):
                for atom_B in range(N_atoms):
                    if atom_A != atom_B:
                        Hessian[
                            3 * atom_A : 3 * atom_A + 3, 3 * atom_A : 3 * atom_A + 3,
                        ] -= Hessian[
                            3 * atom_A : 3 * atom_A + 3, 3 * atom_B : 3 * atom_B + 3,
                        ]

                Hessian[3 * atom_A : 3 * atom_A + 3, 3 * atom_A : 3 * atom_A + 3] = (
                    Hessian[3 * atom_A : 3 * atom_A + 3, 3 * atom_A : 3 * atom_A + 3]
                    + np.transpose(
                        Hessian[3 * atom_A : 3 * atom_A + 3, 3 * atom_A : 3 * atom_A + 3],
                    )
                ) / 2   

            return Hessian

    def predict_array(self,folders):
        for path in folders:
            mol = Molecule(path,self.config.molecule.xyz_file)
            
            if self.config.molecule.feature in ["reduced"]:
                mol.feature = ReducedFeature
            elif self.config.molecule.feature in ["custom","numpy","numpy_list"]:
                mol.feature = CustomFeature

            print(f"Path: {path}")
            print(f"Number of atoms: {mol.nat}")
            
            mol = self.predict(mol)
            
            if mol.calc_succeeded:
                wrt_hess_to_xtb(os.path.join(mol.path, "MLhessian"),mol.ml_hessian.hessian)
                #np.save(os.path.join(mol.path, "MLhessian.npy"),mol.ml_hessian.hessian)

            if mol.nat == 1:
                mol.ml_hessian.hessian = np.zeros([3,3])
                wrt_hess_to_xtb(os.path.join(mol.path, "MLhessian"),mol.ml_hessian.hessian)

    def test_array(self,folders):
        n_val = 0 
        err = 0.0
        s_err = 0.0
        for path in folders:
            mol = Molecule(path,self.config.molecule.xyz_file)
            
            if self.config.molecule.feature in ["reduced"]:
                mol.feature = ReducedFeature
            elif self.config.molecule.feature in ["custom","numpy","numpy_list"]:
                mol.feature = CustomFeature

            print(f"Path: {path}")
            print(f"Number of atoms: {mol.nat}")
            
            mol = self.predict(mol)
        
            mol.read_hessian(os.path.join(mol.path, "hessian"))
            
            err += np.sum(np.abs(mol.ml_hessian.hessian - mol.hessian.hessian))
            s_err += np.sum((mol.ml_hessian.hessian - mol.hessian.hessian)**2)
            n_val += mol.hessian.hessian.shape[0]*mol.hessian.hessian.shape[1]
        
        return np.sqrt(s_err/n_val),err/n_val
                
def restructure_RH(hess_vec_ab, atom_pairs, R_MI_APF_mat, transpose_list):

    hess_ab = np.zeros((len(hess_vec_ab),3,3))
    rabs = np.zeros((len(hess_vec_ab),3,3))
    
    ite_hetero = 0
    for atom_A,atom_B in zip(atom_pairs[:,0],atom_pairs[:,1]): 
        hess_ab[ite_hetero] = hess_vec_ab[ite_hetero].reshape(3,3)
        if transpose_list[ite_hetero]:
            hess_ab[ite_hetero] = hess_ab[ite_hetero].T 

        rabs[ite_hetero] = R_MI_APF_mat[
                3 * atom_A : 3 * atom_A + 3, 3 * atom_B : 3 * atom_B + 3,
            ].T
        ite_hetero += 1

    return rabs,hess_ab

    # def error_estimation(self:Environment, folders, rnd_seed, train_size):
    #     print("Computing error on test set")

    #     for folder in folders:
    #         if folder not in self.not_considered:

    #             mol = Molecule(folder,self.config.molecule.xyz_file)

    #             mol.hessians_difference(self.config.molecule.target_file, "MLhessian")
    #             shape = np.shape(mol.hess_diff)
    #             size += shape[0]**2
    #             error += np.sum(mol.hess_diff**2)

    #     error = np.sqrt(error / size)

    #     print("Seed\tTrain Size\tRMSD")
    #     print(f"{rnd_seed}\t{train_size*100: 3.0f}\t{error : 0.5f}")

    #     with open("results","a+") as file:
    #         file.write(f"{rnd_seed}\t{train_size*100: 3.0f}\t{error : 0.5f}\n")

    # def optimization(self,mol:Molecule):
    #     self.model = load(os.path.join("", "2MDhess.joblib"))

    #     mol.feature.

    #     self.mol.setConfiguration(folder,self.config.general,self.config.molecule)

    #     self.mol.ProcessData(model=self.model)

    #     self.mol.optimize_step()

class HessianPMPredictor(HessianPredictor):

    @checkTiming(enabled=True)
    def predict(self, mol:Molecule):
        """Predict the Hessian of a molecule from its features.

        Args:
            mol (Molecule): Molecule
        """
        if mol.nat == 1:
            mol.calc_succeeded = False

        feat = mol.feature.processed_features.astype(np.float32)
        

        if mol.calc_succeeded:
            upper_tri_blocks_hessian = self.model.predict(feat)
            upper_tri_blocks_hessian[:,:9] += upper_tri_blocks_hessian[:,9:]

            mol.ml_hessian.hessian = self.gen_hess_from_vec_pred_damped(
                upper_tri_blocks_hessian[:,:9],
                mol.nat,
                mol.feature.R_MI_APF_mat,
                mol.feature.transpose_list,
                mol.computed_atom_pairs,
                mol.feature.distance_mat,
            )
            
            del upper_tri_blocks_hessian

        return mol