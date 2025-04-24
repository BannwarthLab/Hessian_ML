from __future__ import annotations

import faulthandler
import os
import subprocess
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd

from scipy.spatial import distance_matrix
from ase.units import Bohr

from copy import deepcopy 

from hess_ml.src2.utilities.decorator import checkTiming
from hess_ml.src2.utilities.matrix_operation import rotate_matrix,rotate_vector_array,rotate_vector
from hess_ml.src2.utilities.parser import parse_dftd4_output

if TYPE_CHECKING:
    from hess_ml.src2.molecule.molecule import Molecule

strings = [
        "response",
        #"gap",
        # "chem_pot",
        # "HOAO_a",
        # "LUAO_a",
        # "HOAO_b",
        # "LUAO_b",
        #"delta_gap",
        # "delta_chem_pot",
        # "delta_HOAO",
        # "delta_LUAO",

        "E_rep",
        "E_EHT",
        "E_disp2",
        "E_disp3",
        "E_ies_ixc",
        "E_AES",
        "E_AXC",
        "E_tot"
    ]

pattern = ""

for string in strings:
    pattern += string + "|"
pattern = pattern[:-1]

class FeatureCalculation:
    def __init__(self,mol:Molecule) -> None:
        self._mol = mol
        self._processed_features: np.ndarray | list | None = None
        self.new_keys: None | list = None 
        return
    
    @property
    def processed_features(self) -> np.ndarray:
        if self._processed_features is None:
            self.get_processed_features()
        return np.array(self._processed_features)
    

    def get_processed_features(self):
        self.ImportFeature()
        self._processed_features = self.scalars.flatten().tolist()
        self._processed_features.extend(self.vectors.flatten().tolist())
        self._processed_features.extend(self.matrices.flatten().tolist())

    def ImportFeature(self):
        
        #try:
        if True:
            faulthandler.enable()
            from tblite.interface import Calculator
            
            calc= Calculator(
            method="GFN2-xTB",
            uhf=self._mol.electronic_properties.uhf,
            charge=self._mol.electronic_properties.charge,
            numbers=np.array(self._mol.atomic_numbers),
            positions=self._mol.xyz*1/Bohr,
            )

            if self._mol.solvent is not None:
                calc.add("alpb-solvation",self._mol.solvent)
                
            calc.set("verbosity", 0)
            
            calc.add("bond-orders")

            calc.add("xtbml.toml")

            res = calc.singlepoint()
            
            X:dict = deepcopy(res.get("post-processing-dict"))

            self.wbo = deepcopy(res.get("bond-orders"))

            self.energy = deepcopy(res.get("energy"))
            
            self.gradient = np.array(deepcopy(res.get("gradient")))

            self.gradient /= Bohr

            X.pop("bond-orders")

            new_keys = self.adapt_keys(X.keys())

            self.ml_feat = pd.DataFrame(deepcopy(X),columns=X.keys())

            #self.ml_feat.to_csv(os.path.join(self._mol.path,'features.csv'))

            self.ml_feat = self.ml_feat.rename(columns=new_keys)
            
            X = None
            res = None
            calc = None

            self._get_dftd4_params()

            self.FilterFeatures()

        # except:  # noqa: E722
        #     self._mol.calc_succeeded = False
        #     calc = None 
        #     res = None 
        #     print("No convergence structure will not be considered.")

    def adapt_keys(self,keys):

        new_keys = {}
        for old_key in keys:
            if "." in old_key:
                new_key = (old_key[:old_key.rindex('_')])
            else:
                new_key = old_key

            new_keys[old_key] = new_key

        return new_keys

    def ReadGradient(self:Molecule, file):
        with open(file, "rb") as f:
            f.close()

        self.gradient = np.genfromtxt(
            file,
            skip_header=2 + self.nat,
            skip_footer=1,
            loose=True,
        )

    def read_wbos(self):
        wboFilePath = os.path.join(self._mol.path, "wbo")
        wbos = np.zeros([self._mol.nat,self._mol.nat])

        with open(wboFilePath) as file:
            lines = file.readlines()
            for line in lines:
                i,j,val = tuple(line.split())
                wbos[int(i)-1,int(j)-1] = float(val)
        file.close()

        wbos += wbos.T

        return wbos
    
    def _get_dftd4_params(self):
        fxyz = os.path.join(self._mol.path,self._mol.fxyz)
        result = subprocess.run(["dftd4", fxyz], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.C6_params = parse_dftd4_output(result.stdout)
    
    def FilterFeatures(self):

        self.dipm = {}
        self.qm = {}
        self.q = {}
        self.cn = {}
        self.p = {}
        self.norms = {}

        self.scalars = []
        self.vectors = []
        self.matrices = []

        self.scalar_keys = []
        self.vector_keys = []
        self.matrix_keys = []

        self.distance_mat = distance_matrix(self._mol.xyz,self._mol.xyz)

        vector = []
        matrix = []
        scalar = []

        for orb in ["s", "p", "d", "A"]:#, "e", "Z"]:

            if orb not in {"s", "p", "d"}:

                self.dipm[f"delta_{orb}"] = self.ml_feat.loc[:,
                    self.ml_feat.columns.str.contains(f"delta_dipm_{orb}_")].to_numpy()

                for idx in range(0,(self.dipm[f"delta_{orb}"]).shape[1],3):
                    vector.append(self.dipm[f"delta_{orb}"][:,idx:idx+3])

                # self.qm[f"delta_{orb}"] = self.ml_feat.loc[:,
                #     self.ml_feat.columns.str.contains(f"delta_qm_{orb}_")].to_numpy()

                #self.qm[f"delta_{orb}"] = self._transform_sym_mat_array(self.qm[f"delta_{orb}"])

                #matrix.append(self.qm[f"delta_{orb}"])

                self.vector_keys.append(f"delta_dipm_{orb}")
                # self.matrix_keys.append(f"delta_qm_{orb}")

                for key in [f"delta_dipm_{orb}"]:#,f"delta_qm_{orb}"
                    temp = self.ml_feat.loc[:,key].to_numpy()
                    scalar.extend(temp.reshape(-1,self._mol.nat))
                    self.scalar_keys.append(key)
                    self.norms[key] = temp

            if orb not in {"e", "Z"}:

                self.dipm[f"{orb}"] = self.ml_feat.loc[:,self.ml_feat.columns.str.startswith(f"dipm_{orb}_")].to_numpy()

                self.qm[f"{orb}"] = self.ml_feat.loc[:,self.ml_feat.columns.str.startswith(f"qm_{orb}_")].to_numpy()

                self.qm[f"{orb}"] = self._transform_sym_mat_array(self.qm[f"{orb}"])

                vector.append(self.dipm[f"{orb}"])
                matrix.append(self.qm[f"{orb}"])

                self.vector_keys.append(f"dipm_{orb}")
                self.matrix_keys.append(f"qm_{orb}")

                if orb != "A":
                    self.p[f"{orb}"] = self.ml_feat.loc[:,self.ml_feat.columns.str.contains(f"p_{orb}")].to_numpy()
                    scalar.append(self.p[f"{orb}"].flatten())
                    self.scalar_keys.append(f"p_{orb}")

                for key in [f"dipm_{orb}",f"qm_{orb}"]:
                    temp = self.ml_feat.loc[:,key].to_numpy()
                    scalar.extend(temp.reshape(-1,self._mol.nat))
                    self.scalar_keys.append(key)
                    self.norms[key] = temp

        self.dipm_norm = np.linalg.norm(self.dipm["A"],axis=1)

        self.energy_based = self.ml_feat.loc[:,self.ml_feat.columns.str.contains(pattern)].to_numpy()
        self.scalar_keys.extend(strings)

        scalar.extend(self.energy_based.T)

        self.cn["default"] = self.ml_feat.loc[:, "CN"].to_numpy()
        self.cn["delta"] = self.ml_feat.loc[:, self.ml_feat.columns.str.startswith("delta_CN")].to_numpy()

        self.scalar_keys.append("default_CN")
        self.scalar_keys.append("delta_CN")

        scalar.append(self.cn["default"])

        scalar.extend(self.cn["delta"].reshape(-1,self._mol.nat))
        
        self.q["default"] = self.ml_feat.loc[:, "q_A"].to_numpy()
        self.q["delta"] = self.ml_feat.loc[:, self.ml_feat.columns.str.contains("delta_q_A")].to_numpy()

        scalar.append(self.q["default"])
        
        scalar.extend(self.q["delta"].reshape(-1,self._mol.nat))

        self.scalar_keys.append("default_q_A")
        self.scalar_keys.append("delta_q_A")

        self.scalars, self.vectors ,self.matrices = self._transform_arrays(np.array(scalar),np.array(vector),np.array(matrix))


    def _transform_arrays(self,scalar:np.ndarray,vector:np.ndarray,matrix:np.ndarray):
        a,b = scalar.shape

        transformed_scalar = np.empty_like(scalar).reshape(b,a)
        a,b,c = vector.shape
        transformed_vector = np.empty_like(vector).reshape(b,a,c)
        a,b,c,d = matrix.shape
        transformed_matrix = np.empty_like(matrix).reshape(b,a,c,d)

        for i in range(self._mol.nat):
            transformed_scalar[i,:] = scalar[:,i]
            transformed_vector[i,:,:] = vector[:,i,:]
            transformed_matrix[i,:,:,:] = matrix[:,i,:,:]

        return transformed_scalar,transformed_vector,transformed_matrix


    def _transform_sym_mat_array(self,mat_array):

        mat_array_new = []
        for atom in range(len(mat_array)):
            temp_mat = np.zeros([3, 3])
            temp_mat[np.tril_indices(temp_mat.shape[0], k=0)] = np.array(mat_array[atom])
            temp_mat = temp_mat + temp_mat.T - np.diag(np.diag(temp_mat))
            mat_array_new.append(temp_mat)

        return np.array(mat_array_new)
    

    def _transform_sym_mat(self,mat):

        temp_mat = np.zeros([3, 3])
        temp_mat[np.tril_indices(temp_mat.shape[0], k=0)] = np.array(mat)
        temp_mat = temp_mat + temp_mat.T - np.diag(np.diag(temp_mat))

        return temp_mat
    

    def gen_Feature_red(self, R_MI_APF, atom_pair:tuple[int,int]) -> tuple:
        """Generate Features for an atom pair.

        Args:
            R_MI_APF (_type_): Rotation matrix
            atom_pair (tuple[int,int]): index of atom A and B

        Returns:
            tuple: Features,transpose, rotation matrix
        """
        Features_temp = []

        A,B = atom_pair

        transpose = None

        # Performs a rotation around the Y axis by 180 ° if nuclear charge of A
        # is smaller than B to achieve a consistent alignment
        # If A == B rotation depends on dipole moment

        if self._mol.atomic_numbers[A] < self._mol.atomic_numbers[B]:
            B, A = A, B

            transpose = [B, A]

            #rotation around the y axis 
            R_MI_APF = -R_MI_APF
            R_MI_APF[1] = -R_MI_APF[1]

        elif self._mol.atomic_numbers[A] == self._mol.atomic_numbers[B]:

            dipm_A_norm = self.dipm_norm[A]
            dipm_B_norm = self.dipm_norm[B]

            if dipm_A_norm < dipm_B_norm:
                B, A = A, B

                transpose = [B, A]

                #rotation around the y axis 
                R_MI_APF = -R_MI_APF
                R_MI_APF[1] = -R_MI_APF[1]

            elif dipm_A_norm == dipm_B_norm:
                print("Nucelar Charge and Dipole moment are the same.")


        xyz_A  = self._mol.xyz[A, :].copy()
        xyz_B = self._mol.xyz[B, :].copy()

        s = 0.5*(xyz_A+xyz_B)

        xyz_A -= s
        xyz_B -= s
        xyz_A = np.matmul(R_MI_APF,xyz_A)
        xyz_B = np.matmul(R_MI_APF,xyz_B)

        r_AB = (xyz_A - xyz_B).reshape(1,-1)

        R_AB = self.distance_mat[A,B]

        Quantity_AB = [[], []]
        etot_idx = self.scalar_keys.index("E_tot")
        #Atom specific information
        for j, atom in enumerate([A, B]):
            # ____Rotation from initial coordinate system to atom pair focused system____

            #This is a 7 x 3 x 3 matrix, rotation of all matrices is achieved with this routine

            quad_moments = rotate_matrix(R_MI_APF,self.matrices[atom])

            #Only takes the lower triangle values as the matrices are symmetric
            Quantity_AB[j].extend(quad_moments[:,np.tril_indices(3)[0],np.tril_indices(3)[1]].flatten().tolist())

            dipole_moments = rotate_vector_array(R_MI_APF,self.vectors[atom])
            Quantity_AB[j].extend(dipole_moments.flatten().tolist())

            # ____Append Features to Feature Vector____

            Quantity_AB[j].extend(self.scalars[atom])
            #Quantity_AB[j].extend(self.scalars[atom]/self.scalars[atom][etot_idx])

        Quantity_AB_arr = np.array(Quantity_AB)

        # Feature_Arith = ((Quantity_AB_arr[0] + Quantity_AB_arr[1]) / 2).tolist()
        Feature_Prod = (Quantity_AB_arr[0] * Quantity_AB_arr[1]).tolist()

        Feature_AbsDiff = (Quantity_AB_arr[0] - Quantity_AB_arr[1]).tolist()

        Features_temp.extend(Quantity_AB[0])
        Features_temp.extend(Quantity_AB[1])

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
        qm_A = rotate_matrix(R_MI_APF,qm_A)

        qm_B = self.qm[qm_key][B]

        qm_B = rotate_matrix(R_MI_APF,qm_B)
        
        order2_aes = q_A*np.matmul(r_AB,np.matmul(qm_B,r_AB.T))
        order2_aes += q_B*np.matmul(r_AB,np.matmul(qm_A,r_AB.T))
        order2_aes -= 3*np.dot(dipm_A,r_AB.T)*np.dot(dipm_B,r_AB.T)
        order2_aes += R_AB**2*np.dot(dipm_A,dipm_B.T)

        order2_aes /= R_AB**5

        C6_A = float(self.C6_params[A])
        C6_B = float(self.C6_params[B])

        potE = q_A*q_B/R_AB
        
        atoms = [A,B]
        for atom in atoms:
            mask = np.ones(self._mol.nat,dtype=bool)
            mask[atom] = False
            V_J = np.sum(self.q["default"][mask]/self.distance_mat[atom,mask])
            Features_temp.append(V_J*self.q["default"][atom])

        Features_temp.append(potE)

        Features_temp.append(C6_A)
        Features_temp.append(C6_B)

        Features_temp.append(self.wbo[A,B])

        Features_temp.extend(order1_aes[0])
        Features_temp.extend(order2_aes[0])

        # Features_temp.extend(Feature_Arith)
        Features_temp.extend(Feature_Prod)
        Features_temp.extend(Feature_AbsDiff)
        
        Features_temp.extend(R_AB**np.array([-12,6,1,-1,-3,-6,-12]))

        return np.array(Features_temp),transpose,R_MI_APF
    
    def gen_Feature(self, R_MI_APF, atom_pair:tuple[int,int]) -> tuple:
        """Generate Features for an atom pair.

        Args:
            R_MI_APF (_type_): Rotation matrix
            atom_pair (tuple[int,int]): index of atom A and B

        Returns:
            tuple: Features,transpose, rotation matrix
        """        
        Features_temp = []

        A,B = atom_pair

        transpose = None

        # Performs a rotation around the Y axis by 180 ° if nuclear charge of A
        # is smaller than B to achieve a consistent alignment
        # If A == B rotation depends on dipole moment

        if self._mol.atomic_numbers[A] < self._mol.atomic_numbers[B]:
            B, A = A, B

            transpose = [B, A]

            R_y= -np.eye(3)
            R_y[1,1] = 1

            R_MI_APF = np.matmul(R_y, R_MI_APF)

        elif self._mol.atomic_numbers[A] == self._mol.atomic_numbers[B]:

            dipm_norm_A = np.linalg.norm(self.dipm["A"][A]) 
            dipm_norm_B = np.linalg.norm(self.dipm["A"][B])

            if dipm_norm_A < dipm_norm_B:
                B, A = A, B

                transpose = [B, A]

                R_y =  -np.eye(3)
                R_y[1,1] = 1

                R_MI_APF = np.matmul(R_y, R_MI_APF)

            elif dipm_norm_A == dipm_norm_B:
                print("Nucelar Charge and Dipole moment are the same.")

        r_AB = (self._mol.xyz[A, :].copy() - self._mol.xyz[B, :].copy()).reshape(1,-1)

        r_AB = np.matmul(R_MI_APF,r_AB.T).T

        R_AB = self.distance_mat[A,B]

        Quantity_AB = [[], []]

        #Atom specific information
        for j, atom in enumerate([A, B]):
            # ____Rotation from initial coordinate system to atom pair focused system____

            grad:np.ndarray = np.matmul(R_MI_APF, self.gradient[atom])
            
            #This is a 7 x 3 x 3 matrix, rotation of all matrices is achieved with this routine

            quad_moments = rotate_matrix(R_MI_APF,self.matrices[atom])

            #Only takes the lower triangle values as the matrices are symmetric
            red_quad_mom = (quad_moments[:,np.tril_indices(3)[0],np.tril_indices(3)[1]].flatten().tolist())
            

            Quantity_AB[j].extend(red_quad_mom)
            # print(f"Quad. Mom. Length: {len(red_quad_mom)}")
            dipole_moments = rotate_vector_array(R_MI_APF,np.array(self.vectors[atom]))
            # print(f"Dip. Mom. Length: {len(dipole_moments.flatten())}")
            Quantity_AB[j].extend(dipole_moments.flatten().tolist())

            # ____Append Features to Feature Vector____
            # print(f"grad Length: {len(grad.tolist())}")
            # print(f"Scalar Length: {len(self.scalars[atom])}")
            # print(f"At. Num. Length: {len([self._mol.atomic_numbers[atom]])}")

            Quantity_AB[j].extend(grad.tolist())
            Quantity_AB[j].extend(self.scalars[atom])
            Quantity_AB[j].extend([self._mol.atomic_numbers[atom]])


        Quantity_AB_arr = np.array(Quantity_AB)

        Feature_Arith = ((Quantity_AB_arr[0] + Quantity_AB_arr[1]) / 2).tolist()
        Feature_Prod = (Quantity_AB_arr[0] * Quantity_AB_arr[1]).tolist()
        Feature_AbsDiff = (Quantity_AB_arr[0] - Quantity_AB_arr[1]).tolist()
        # print(f"Tot. Quant A or B Length: {len(Quantity_AB[0])}")
        Features_temp.extend(Quantity_AB[0])
        Features_temp.extend(Quantity_AB[1])
        # print("Feature length temp",len(Features_temp))
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
        qm_A = rotate_matrix(R_MI_APF,qm_A)
        
        qm_B = self.qm[qm_key][B]

        qm_B = rotate_matrix(R_MI_APF,qm_B)
        
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
        # wbo_r_norm = 0.0
        # loc_nuc_charge_norm = 0.0

        for atom in atoms:
            wbo_r = np.zeros(3)
            nuc_charge_loc = np.zeros(3)
            n_adj =  0
            
            for idx in range(self._mol.nat):
                if idx not in atoms and self.wbo[atom,idx] > wbo_th:
                    n_adj += 1
                    temp_r_ab = self._mol.xyz[idx].copy()-self._mol.xyz[atom].copy()
                    r_ab_norm = self.distance_mat[idx,atom]#np.linalg.norm(temp_r_ab)

                    wbo_r += self.wbo[atom,idx]/r_ab_norm**3 * np.matmul(R_MI_APF,temp_r_ab)
                    nuc_charge_loc += self._mol.atomic_numbers[idx]*self._mol.atomic_numbers[atom]/r_ab_norm**3 * np.matmul(R_MI_APF,temp_r_ab)
                    #wbo_r_norm += np.linalg.norm(wbo_r)
                    #loc_nuc_charge_norm += np.linalg.norm(nuc_charge_loc)
            

            mask = np.ones(self._mol.nat,dtype=bool)
            
            mask[atom] = False

            V_J = np.sum(self.q["default"][mask]/self.distance_mat[atom,mask])


            # print("VJ")
            Features_temp.append(V_J*self.q["default"][atom])
            # print("adj. Atoms")
            Features_temp.append(n_adj)
            
            # print("wbo_r",len(wbo_r))
            Features_temp.extend(wbo_r)
            #Features_temp.append(wbo_r_norm)
            # print("nuc charge loc. Atoms",len(nuc_charge_loc))
            Features_temp.extend(nuc_charge_loc)
            #Features_temp.append(loc_nuc_charge_norm)

        # print("pot E ")
        Features_temp.append(potE)

        # print("C6A")
        Features_temp.append(C6_A)
        # print("C6B")
        Features_temp.append(C6_B)
        # print("wbo")
        Features_temp.append(self.wbo[A,B])
        # print("r_AB",len(r_AB.tolist()[0]))
        Features_temp.extend(r_AB.tolist()[0])
        # print("aes ord1 ord2")
        Features_temp.extend(order1_aes[0])
        Features_temp.extend(order2_aes[0])


        # print("Arith,Prod,Diff")
        Features_temp.extend(Feature_Arith)
        Features_temp.extend(Feature_Prod)
        Features_temp.extend(Feature_AbsDiff)
        
        # print("RAB")

        for i in [12,6,1,-1,-2,-3,-6]:
            Features_temp.extend([R_AB**i])

        return np.array(Features_temp),transpose,R_MI_APF

    def gen_Feature_custom(self, R_MI_APF, atom_pair:tuple[int,int]) -> tuple:
        """Generate Features for an atom pair.

        Args:
            R_MI_APF (_type_): Rotation matrix
            atom_pair (tuple[int,int]): index of atom A and B

        Returns:
            tuple: Features,transpose, rotation matrix
        """
        Features_temp = []

        A,B = atom_pair

        transpose = None

        # Performs a rotation around the Y axis by 180 ° if nuclear charge of A
        # is smaller than B to achieve a consistent alignment
        # If A == B rotation depends on dipole moment

        if self._mol.atomic_numbers[A] < self._mol.atomic_numbers[B]:
            B, A = A, B

            transpose = [B, A]

            #rotation around the y axis 
            R_MI_APF = -R_MI_APF
            R_MI_APF[1] = -R_MI_APF[1]

        elif self._mol.atomic_numbers[A] == self._mol.atomic_numbers[B]:

            dipm_A_norm = self.dipm_norm[A]
            dipm_B_norm = self.dipm_norm[B]

            if dipm_A_norm < dipm_B_norm:
                B, A = A, B

                transpose = [B, A]

                #rotation around the y axis 
                R_MI_APF = -R_MI_APF
                R_MI_APF[1] = -R_MI_APF[1]

            elif dipm_A_norm == dipm_B_norm:
                print("Nucelar Charge and Dipole moment are the same.")


        xyz_A  = self._mol.xyz[A, :].copy()
        xyz_B = self._mol.xyz[B, :].copy()

        s = 0.5*(xyz_A+xyz_B)

        xyz_A -= s
        xyz_B -= s
        xyz_A = np.matmul(R_MI_APF,xyz_A)
        xyz_B = np.matmul(R_MI_APF,xyz_B)

        r_AB = (xyz_A - xyz_B).reshape(1,-1)

        R_AB = self.distance_mat[A,B]

        Quantity_AB = [[], []]
        etot_idx = self.scalar_keys.index("E_tot")
        #Atom specific information

        for j, atom in enumerate([A, B]):
            # ____Rotation from initial coordinate system to atom pair focused system____
            
            #This is a 7 x 3 x 3 matrix, rotation of all matrices is achieved with this routine

            quad_moments = rotate_matrix(R_MI_APF,self.matrices[atom])

            #Only takes the lower triangle values as the matrices are symmetric
            Quantity_AB[j].extend(quad_moments[:,np.tril_indices(3)[0],np.tril_indices(3)[1]].flatten().tolist())

            dipole_moments = rotate_vector_array(R_MI_APF,self.vectors[atom])
            Quantity_AB[j].extend(dipole_moments.flatten().tolist())

            # ____Append Features to Feature Vector____

            Quantity_AB[j].extend(self.scalars[atom])
            Quantity_AB[j].extend(self.scalars[atom]/self.scalars[atom][etot_idx])

        Quantity_AB_arr = np.array(Quantity_AB)

        # Feature_Arith = ((Quantity_AB_arr[0] + Quantity_AB_arr[1]) / 2).tolist()
        Feature_Prod = (Quantity_AB_arr[0] * Quantity_AB_arr[1]).tolist()

        Feature_AbsDiff = (Quantity_AB_arr[0] - Quantity_AB_arr[1]).tolist()

        Features_temp.extend(Quantity_AB[0])
        Features_temp.extend(Quantity_AB[1])

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
        qm_A = rotate_matrix(R_MI_APF,qm_A)
        
        qm_B = self.qm[qm_key][B]

        qm_B = rotate_matrix(R_MI_APF,qm_B)
        
        order2_aes = q_A*np.matmul(r_AB,np.matmul(qm_B,r_AB.T))
        order2_aes += q_B*np.matmul(r_AB,np.matmul(qm_A,r_AB.T))
        order2_aes -= 3*np.dot(dipm_A,r_AB.T)*np.dot(dipm_B,r_AB.T)
        order2_aes += R_AB**2*np.dot(dipm_A,dipm_B.T)

        order2_aes /= R_AB**5

        C6_A = float(self.C6_params[A])
        C6_B = float(self.C6_params[B])

        potE = q_A*q_B/R_AB
        
        # atoms = [A,B]
        # for atom in atoms:
        #     mask = np.ones(self._mol.nat,dtype=bool)
        #     mask[atom] = False
        #     V_J = np.sum(self.q["default"][mask]/self.distance_mat[atom,mask])
        #    Features_temp.append(V_J*self.q["default"][atom])

        Features_temp.append(potE)
        Features_temp.append(C6_A)
        Features_temp.append(C6_B)

        Features_temp.append(self.wbo[A,B])
        Features_temp.extend(order1_aes[0])
        Features_temp.extend(order2_aes[0])

        #Features_temp.extend(Feature_Arith)
        
        Features_temp.extend(Feature_Prod)
        Features_temp.extend(Feature_AbsDiff)
        
        Features_temp.extend(R_AB**np.array([-12,6,1,-1,-3,-6,-12]))

        return np.array(Features_temp),transpose,R_MI_APF
    
    def get_start_specific_key(self,keys:list[str],starting_string:str):

        for key in keys:
            if key.startswith(starting_string):
                print(key)
                break

        return key

    def supporting_vector(self,atom_pair):
        i,j = atom_pair

        support_vec = (self._mol.feature.dipm["A"][i] + self._mol.feature.dipm["A"][j])

        #support_vec = np.cross(self._mol.feature.dipm["A"][i],self._mol.feature.dipm["A"][j])

        if np.sum(np.abs(support_vec))/3 < 1e-5:
            support_vec = np.array([0.0,1.0,0.0])

        return support_vec