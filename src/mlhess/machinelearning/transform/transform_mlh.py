from __future__ import annotations
import sys 
import numpy as np

from typing import TYPE_CHECKING
from copy import deepcopy

from multiprocessing import shared_memory
from joblib import Parallel, delayed

from mlhess.machinelearning.feature.processing_schemes import new_gen_pair_features
from mlhess.utils.math.matrix_operation import get_rotated_33_block_matrix
from mlhess.machinelearning.target.hessian import NuclearHessianPM
import time as time

if TYPE_CHECKING:
    from mlhess.utils.chemistry.molecule import Molecule


# def init_worker(mol: Molecule):
#     global GLOBAL_MOL
#     GLOBAL_MOL = mol  # type: ignore[name-defined]
def init_worker(meta_list):
    global GLOBAL_ARRAYS, GLOBAL_SHMS

   # if num_threads > 1:
    GLOBAL_ARRAYS = []
    GLOBAL_SHMS = []

    for name, shape, dtype in meta_list:
        shm = shared_memory.SharedMemory(name=name)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        GLOBAL_ARRAYS.append(arr)
        GLOBAL_SHMS.append(shm)

def set_shared_memory_array(self):
    # your data
    dipm_key = "A"
    
    atomic_numbers = np.array(self.atomic_numbers)
    dipm_norm = np.array(self.feature.dipm_norm)
    xyz = self.xyz
    distance_mat = np.array(self.feature.distance_mat)
    q = np.array(self.feature.q['default'])
    dipm = np.array(self.feature.dipm[dipm_key])
    qm_key = "A"
    qm = np.array(self.feature.qm[qm_key])
    scalars = self.feature.scalars
    vectors = self.feature.vectors
    matrices = self.feature.matrices
    etot_idx = np.array([self.scalar_keys.index("E_tot")])#np.array(self.feature.scalar_keys)
    C6_params = self.feature.C6_params
    
    wbo = self.feature.wbo
    arrays = atomic_numbers,dipm_norm,xyz,distance_mat,q,dipm,qm,scalars,vectors,matrices,etot_idx,C6_params,wbo
    return arrays

def init_memory(arrays):
    meta = []
    shms = []
    for arr in arrays:
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        #resource_tracker.unregister(shm.name, 'shared_memory')
        shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        shared_arr[:] = arr[:]

        meta.append((shm.name, arr.shape, arr.dtype))
        shms.append(shm)
    return meta,shms
    

# def transform_prediction(self: Molecule, num_threads):
#     """Processes the features to ML readable."""
#     if self.nat == 1:
#         self.calc_succeeded = False

#     if self.calc_succeeded:
#         temp_dist_mat = self.feature.distance_mat.copy()
#         temp_dist_mat[np.tril_indices_from(temp_dist_mat)] = np.inf
#         atom_pairs:list = list(np.argwhere(temp_dist_mat < 20))

#         self.computed_atom_pairs = atom_pairs

#         init_worker(self)
#         with Pool(processes=num_threads) as pool:
#             results_iterator = pool.map(_transform_block_prediction, atom_pairs) 
#             pool.terminate()
#             pool.join()
#         pool.close()

#         R_MI_APFs, processed_feature, transposes = zip(*deepcopy(results_iterator))

#         self.feature.processed = processed_feature

#         transposes = list(transposes) # type: ignore

#         for val in transposes:
#             if val is not None:
#                 self.transpose_list.append(val)

#         for atom_pair, rot_mat in zip(atom_pairs, R_MI_APFs):
#             atom_A, atom_B = atom_pair
#             i0 = 3 * atom_A
#             i3 = 3 * atom_A + 3
#             j0 = 3 * atom_B
#             j3 = 3 * atom_B + 3

#             self.R_MI_APF_mat[i0:i3, j0:j3] = rot_mat


# def _transform_training(mol: Molecule, num_threads: int = 4):
#     """Transforms the Hessian matrix and features of a molecule into a target-descriptor relation for training."""

#     if mol.nat == 1:
#         mol.calc_succeeded = False

#     if mol.calc_succeeded:
#         mol.transpose_list = []

#         temp_dist_mat = mol.feature.distance_mat.copy()
#         temp_dist_mat[np.tril_indices_from(temp_dist_mat)] = np.inf
#         atom_pairs:list = list(np.argwhere(temp_dist_mat < 20))

#         mol.computed_atom_pairs = atom_pairs

#         init_worker(mol)
#         with Pool(processes=num_threads) as pool:
#             results_iterator = pool.map(_transform_block_training, atom_pairs) 
#             pool.terminate()
#             pool.join()
#         pool.close()

#         mol.feature.processed, mol.processed_target, transposes = process_results(   # type: ignore
#             results_iterator
#         )

#         for val in transposes:
#             if val is not None:
#                 mol.transpose_list.append(val)

#         del atom_pairs, pool, results_iterator

#     mol.feature.processed = np.array(mol.feature.processed)


# def _transform_block_prediction(atom_pair:tuple[int,int]) -> tuple:
#     """Construction of feature vector for the prediction of the an AB Hessian matrix block.

#     Args:
#         index (int): index of for the list of atom pairs.
#         atom_pairs (list[tuple]): list of atom pairs for the comp. of the rotation matrix.

#     Returns:
#         tuple: Rotation Matrix, Features, transpose info
#     """
#     mol = GLOBAL_MOL  # type: ignore[name-defined] #noQA: F821 
#     sup_vector = supporting_vector(mol.feature.dipm["A"], atom_pair)

#     R_MI_APF = get_atom_pair_rot_mat(mol.xyz, sup_vector, atom_pair)

#     Feature_AB, transpose, R_MI_APF = gen_pair_features(
#         mol.feature,
#         R_MI_APF,
#         atom_pair,  # type: ignore[arg-type]
#     )

#     return R_MI_APF, Feature_AB, transpose


# def _transform_block_training(atom_pair: tuple[int,int]) -> tuple:
#     """Rotation of an AB Hessian matrix block and the construction its feature vector.

#     :param index: index of for the list of atom pairs
#     :param atom_pairs: list of atom pairs for the comp. of the rotation matrix"""
#     mol = GLOBAL_MOL  # type: ignore[name-defined] #noQA: F821 

#     atom_A, atom_B = atom_pair
#     sup_vector = supporting_vector(mol.feature.dipm["A"], atom_pair)

#     R_MI_APF = get_atom_pair_rot_mat(
#         mol.xyz,
#         sup_vector,
#         atom_pair,
#     )

#     Feature_AB, transpose, R_MI_APF = gen_pair_features(
#         mol.feature,
#         R_MI_APF,  # type: ignore[arg-type]
#         atom_pair,  # type: ignore[arg-type]
#     )

#     H_APF = mol.hessian.get_rotated_matrix(R_MI_APF, atom_A, atom_B, transpose)

#     return H_APF.flatten(), Feature_AB, transpose


def process_results(results:list) -> tuple[list, list, list]:
    # Process and release memory for results incrementally
    processed_features: list[np.ndarray] = []
    processed_target: list[np.ndarray] = []

    processed_target.extend(result[0] for result in results)
    processed_features.extend(result[1] for result in results)
    transposes: list[list | None] = [result[2] for result in results]
        # Release memory for results
    del results

    return processed_features, processed_target, transposes

GLOBAL_ARRAYS :np.ndarray |None = None
GLOBAL_SHMS:np.ndarray |None  = None
GLOBAL_HESS:np.ndarray |None  = None 
GLOBAL_SHM_HESS:np.ndarray |None  = None 

def attach_shared_memory(meta):
    global GLOBAL_ARRAYS, GLOBAL_SHMS
    if GLOBAL_ARRAYS is None:
        GLOBAL_ARRAYS = []
        GLOBAL_SHMS = []
        if sys.platform == "darwin":
            GLOBAL_ARRAYS.extend(meta)
            GLOBAL_SHMS = None
        else:
            for name, shape, dtype in meta:
                shm = shared_memory.SharedMemory(name=name)
                arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                GLOBAL_ARRAYS.append(arr)
                GLOBAL_SHMS.append(shm)

def detach_shared_memory():
    global GLOBAL_SHMS, GLOBAL_ARRAYS
    
    if GLOBAL_SHMS is not None:
        for shm in GLOBAL_SHMS:
            shm.close()   
        
    GLOBAL_SHMS = None
    GLOBAL_ARRAYS = None


def attach_shared_memory_hessian(meta):
    global GLOBAL_HESS, GLOBAL_SHM_HESS
    if sys.platform == "darwin":
        # Single-thread fallback: meta contains actual array
        GLOBAL_HESS = meta[0]   # just pass ndarray directly
        GLOBAL_SHM_HESS = None
        return meta, None
    else:
        name, shape, dtype = meta[0]
        shm = shared_memory.SharedMemory(name=name)
        GLOBAL_HESS = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        GLOBAL_SHM_HESS = shm
        return meta,shm 

def detach_shared_memory_hessian():
    global GLOBAL_HESS, GLOBAL_SHM_HESS
    
    if GLOBAL_SHM_HESS is not None:
        shm = GLOBAL_SHM_HESS
        shm.close()   

    GLOBAL_HESS = None 
    GLOBAL_SHM_HESS = None 
    return 

def transform_prediction(self: Molecule, num_threads):
    """Processes the features to ML readable."""

    if self.nat == 1:
        self.calc_succeeded = False

    if self.calc_succeeded:

        temp_dist_mat = self.feature.distance_mat.copy()
        temp_dist_mat[np.tril_indices_from(temp_dist_mat)] = np.inf
        atom_pairs:list = list(np.argwhere(temp_dist_mat < 20))
        self.computed_atom_pairs = atom_pairs

        
        arrays = set_shared_memory_array(self)

        if sys.platform =="darwin":
            backend = 'threading'
            meta = arrays
            shms =[]
        else:
            meta,shms = init_memory(arrays)
            backend = 'loky'

        parallel = Parallel(n_jobs=num_threads, backend=backend)
        output_generator = parallel(delayed(_transform_block_prediction)(atom_pair,meta) for atom_pair in atom_pairs)
        R_MI_APFs, processed_feature, transposes = zip(*deepcopy(output_generator))

        self.feature.processed = processed_feature

        transposes = list(transposes) # type: ignore

        for val in transposes:
            if val is not None:
                self.transpose_list.append(val)

        for atom_pair, rot_mat in zip(atom_pairs, R_MI_APFs):
            atom_A, atom_B = atom_pair
            i0 = 3 * atom_A
            i3 = 3 * atom_A + 3
            j0 = 3 * atom_B
            j3 = 3 * atom_B + 3

            self.R_MI_APF_mat[i0:i3, j0:j3] = rot_mat

        for shm in shms:
            shm.close()
            shm.unlink()


def transform_training(mol: Molecule, num_threads: int = 4):
    """Transforms the Hessian matrix and features of a molecule into a target-descriptor relation for training."""
 
    if mol.nat == 1:
        mol.calc_succeeded = False

    if mol.calc_succeeded:
        mol.transpose_list = []

        temp_dist_mat = mol.feature.distance_mat.copy()
        temp_dist_mat[np.tril_indices_from(temp_dist_mat)] = np.inf
        atom_pairs:list = list(np.argwhere(temp_dist_mat < 20))

        mol.computed_atom_pairs = atom_pairs

        arrays = set_shared_memory_array(mol)


        if isinstance(mol.hessian,NuclearHessianPM):
            a,b = mol.hessian.hessian.shape
            hessian = np.zeros((2,a,b))

            hessian[0,:,:] = mol.hessian.hess_p
            hessian[1,:,:] = mol.hessian.hess_m

        else:
            hessian = mol.hessian.hessian


        if sys.platform =="darwin":
            backend = 'threading'
            meta = arrays
            meta_hess = [hessian]
            shms,shm_hess =[],[]
        else:
            meta,shms = init_memory(arrays)
            meta_hess,shm_hess =init_memory([hessian])
            backend = 'loky'

        
        with Parallel(n_jobs=num_threads, backend=backend, prefer="processes") as parallel:
            output_generator = parallel(delayed(_transform_block_training)(atom_pair,meta,meta_hess) for atom_pair in atom_pairs)
            parallel._backend.terminate()

        mol.feature.processed, mol.processed_target, transposes = process_results(   # type: ignore
            output_generator
        )

        for val in transposes:
            if val is not None:
                mol.transpose_list.append(val)

        del atom_pairs, output_generator

        for shm in shms:
            shm.close()
            shm.unlink()

        for shm in shm_hess:
            shm.close()
            shm.unlink()

        del arrays
        del hessian             

    mol.feature.processed = np.array(mol.feature.processed)

def _transform_block_training(atom_pair: tuple[int,int],meta,meta_hess) -> tuple:
    """Rotation of an AB Hessian matrix block and the construction its feature vector.

    :param index: index of for the list of atom pairs
    :param atom_pairs: list of atom pairs for the comp. of the rotation matrix"""

    attach_shared_memory(meta)
    attach_shared_memory_hessian(meta_hess)

    try:
        atom_A, atom_B = atom_pair
        Feature_AB, transpose, R_MI_APF = new_gen_pair_features(atom_pair,GLOBAL_ARRAYS)
        if len(GLOBAL_HESS.shape)==3: #type: ignore 
            H_APF_p = get_rotated_33_block_matrix(R_MI_APF,GLOBAL_HESS[0], atom_A, atom_B, transpose) #type: ignore 
            H_APF_m = get_rotated_33_block_matrix(R_MI_APF,GLOBAL_HESS[1], atom_A, atom_B, transpose) #type: ignore 
            H_APF = np.array([H_APF_p,H_APF_m])
        else:        
            H_APF = get_rotated_33_block_matrix(R_MI_APF,GLOBAL_HESS, atom_A, atom_B, transpose) #type: ignore 

    finally:
        detach_shared_memory() 
        detach_shared_memory_hessian()

    return H_APF.flatten(), Feature_AB, transpose

def _transform_block_prediction(atom_pair:tuple[int,int],meta) -> tuple:
    """Construction of feature vector for the prediction of the an AB Hessian matrix block.

    Args:
        index (int): index of for the list of atom pairs.
        atom_pairs (list[tuple]): list of atom pairs for the comp. of the rotation matrix.

    Returns:
        tuple: Rotation Matrix, Features, transpose info
    """

    attach_shared_memory(meta)
    try:
        Feature_AB, transpose, R_MI_APF = new_gen_pair_features(atom_pair,GLOBAL_ARRAYS)
    finally:
        detach_shared_memory() 

    return R_MI_APF, Feature_AB, transpose

