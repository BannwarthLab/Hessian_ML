import time
from multiprocessing import Pool
import numpy as np

from functools import partial 

from hess_ml.src.decorator.decorator import checkTiming
from hess_ml.src.geometry import Geometry
from hess_ml.src.rotation_func import Rotation_Functions

class FeatureGen(Geometry):
    @checkTiming(enabled=False)
    def gen_Feature(self, R_MI_APF, atom_A: int, atom_B: int) -> tuple:
        Features_temp = []

        A = atom_A
        B = atom_B

        transpose = None

        # Performs a rotation around the X axis by 180 ° if nuclear charge of A
        # is smaller than B to achieve a consistent alignment
        # If A == B rotation depends on dipole moment

        if self.NuclearCharge[A] < self.NuclearCharge[B]:
            B, A = A, B

            transpose = [B, A]

            rot_Mat_X = self.rot_X(np.pi)

            R_MI_APF = np.matmul(rot_Mat_X, R_MI_APF)

            rot_Mat_Z = self.rot_Z(np.pi)

            R_MI_APF = np.matmul(rot_Mat_Z, R_MI_APF)

        elif self.NuclearCharge[A] == self.NuclearCharge[B]:
            if np.linalg.norm(self.dipm["A"][A]) < np.linalg.norm(self.dipm["A"][B]):
                B, A = A, B

                transpose = [B, A]

                rot_Mat = self.rot_X(np.pi)

                R_MI_APF = np.matmul(rot_Mat, R_MI_APF)

                rot_Mat = self.rot_Z(np.pi)

                R_MI_APF = np.matmul(rot_Mat, R_MI_APF)

            elif np.linalg.norm(self.dipm["A"][A]) == np.linalg.norm(self.dipm["A"][B]):
                print("Nucelar Charge and Dipole moment are the same.")


        r_AB = (self.xyz[A, :] - self.xyz[B, :]).reshape(1,-1)

        r_AB = np.matmul(R_MI_APF,r_AB.T).T

        R_AB = np.linalg.norm(self.xyz[A, :] - self.xyz[B, :])

        Quantity_AB = [[], []]

        #Atom specific information 
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
                #Quantity_AB[j].extend([np.linalg.norm(temp_qm)])

            for dipm_key in self.dipm:

                temp_dipm:np.ndarray = np.matmul(R_MI_APF, self.dipm[dipm_key][atom])
                Quantity_AB[j].extend(temp_dipm.tolist())
                #Quantity_AB[j].extend([np.linalg.norm(temp_dipm)])

            for p_key in self.p:
                Quantity_AB[j].extend(self.p[p_key][atom].tolist())

            # ____Append Features to Feature Vector____

            Quantity_AB[j].extend(grad.tolist())
            Quantity_AB[j].extend(self.energy_based[atom].tolist())

            Quantity_AB[j].extend([self.NuclearCharge[atom]])

            Quantity_AB[j].extend([self.cn["default"][atom]])
            Quantity_AB[j].extend(self.cn["delta"][atom])

            Quantity_AB[j].extend([self.q["default"][atom]])
            Quantity_AB[j].extend(self.q["delta"][atom])

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

        qm_A = np.zeros([3, 3])
        qm_A[np.tril_indices(qm_A.shape[0], k=0)] = self.qm[qm_key][A]
        qm_A = qm_A + qm_A.T - np.diag(np.diag(qm_A))

        qm_A = np.matmul(
                    np.matmul(R_MI_APF, qm_A),
                    np.transpose(R_MI_APF),
                )

        qm_B = np.zeros([3, 3])
        qm_B[np.tril_indices(qm_B.shape[0], k=0)] = self.qm[qm_key][B]
        qm_B = qm_B + qm_B.T - np.diag(np.diag(qm_B))

        qm_B = np.matmul(
                    np.matmul(R_MI_APF, qm_B),
                    np.transpose(R_MI_APF),
                )
        
        order2_aes = q_A*np.matmul(r_AB,np.matmul(qm_B,r_AB.T))
        order2_aes += q_B*np.matmul(r_AB,np.matmul(qm_A,r_AB.T))

        order2_aes -= 3*q_B*np.dot(dipm_A,r_AB.T)*np.dot(dipm_B,r_AB.T)
        order2_aes += R_AB**2*np.dot(dipm_A,dipm_B.T)

        order2_aes /= R_AB**5

        C6_A = self.C6_params[A]
        C6_B = self.C6_params[B]

        C8_A = self.C8_params[A]
        C8_B = self.C8_params[B]

        potE = q_A*q_B/R_AB

        Features_temp.append(potE)

        Features_temp.append(C6_A)
        Features_temp.append(C6_B)

        Features_temp.append(C8_A)
        Features_temp.append(C8_B)

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

        return np.array(Features_temp),transpose



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
    def Transform_np(self):
        if self.do_calc:
            self.R_MI_APF_mat = np.zeros([self.N_atoms * 3, self.N_atoms * 3])
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
                    feature,transpose = self.gen_Feature(R_MI_APF, atom_A, atom_B)

                    if transpose is not None:
                        self.transpose_list.append(transpose)

                    self.Feature_AB.append(feature)



    @checkTiming(enabled=True)
    def Transform(self):
        if self.do_calc:
            num_cpus = 4
            self.Feature_AB = []
            self.R_MI_APF_mat = np.zeros([self.N_atoms * 3, self.N_atoms * 3])
            self.transpose_list = []
            atoms = []
            for atom_A in range(self.N_atoms):
                for atom_B in range(atom_A + 1, self.N_atoms):
                    atoms.append((atom_A,atom_B))                


            partial_func = partial(self.single_transform,atoms=atoms)

            indices = [i for i in range(len(atoms))]

            with Pool(processes=num_cpus) as pool:
                results = pool.map(partial_func,indices)

            R_MI_APFs,self.Feature_AB,transposes = zip(*results)

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
    
    def single_transform(self,index:int,atoms:list):
        
        atom_A,atom_B = atoms[index]
        xyz = self.xyz.copy()

        R_MI_APF = self.get_R_euler(
            xyz,
            self.dipm["A"],
            atom_A,
            atom_B,
        )

        Feature_AB,transpose = self.gen_Feature(R_MI_APF, atom_A, atom_B)

        return R_MI_APF,Feature_AB,transpose


class TransformTrain(Rotation_Functions, FeatureGen):
    @checkTiming(enabled=True)
    def Transform_np(self):
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

                    feature,transpose = self.gen_Feature(R_MI_APF, atom_A, atom_B)

                    if transpose is not None:
                        self.transpose_list.append(transpose)

                    self.Feature_AB.append(feature)

                    i0 = 3 * atom_A
                    i3 = 3 * atom_A + 3
                    j0 = 3 * atom_B
                    j3 = 3 * atom_B + 3

                    H_APF = np.matmul(
                        np.matmul(R_MI_APF, self.target[i0:i3, j0:j3]),
                        np.transpose(R_MI_APF),
                    )  # Change Hessian

                    if transpose is not None:
                        H_APF = np.matmul(
                            np.matmul(self.rot_X(np.pi), np.transpose(H_APF)),
                            np.transpose(self.rot_X(np.pi)),
                        )
                        H_APF = np.matmul(
                            np.matmul(self.rot_Z(np.pi), (H_APF)),
                            np.transpose(self.rot_Z(np.pi)),
                        )

                    self.Target_AB.append(list(H_APF.flatten()))


    @checkTiming(enabled=True)
    def Transform(self):

        self.Feature_AB = []
        self.Target_AB = []

        if self.do_calc:
            num_cpus = 4
            self.transpose_list = []
            atoms = []
            for atom_A in range(self.N_atoms):
                for atom_B in range(atom_A + 1, self.N_atoms):
                    atoms.append((atom_A,atom_B))                

            partial_func = partial(self.single_transform,atoms=atoms)

            indices = [i for i in range(len(atoms))]

            with Pool(processes=num_cpus) as pool:
                results = pool.map(partial_func,indices)

            self.Target_AB,self.Feature_AB,transposes = zip(*results)

            self.Feature_AB = list(self.Feature_AB)
            self.Target_AB = list(self.Target_AB)

            for val in transposes:
                if val is not None:
                    self.transpose_list.append(val)
    
    def single_transform(self,index:int,atoms:list):
        
        atom_A,atom_B = atoms[index]
        xyz = self.xyz.copy()

        R_MI_APF = self.get_R_euler(
            xyz,
            self.dipm["A"],
            atom_A,
            atom_B,
        )

        Feature_AB,transpose = self.gen_Feature(R_MI_APF, atom_A, atom_B)

        i0 = 3 * atom_A
        i3 = 3 * atom_A + 3
        j0 = 3 * atom_B
        j3 = 3 * atom_B + 3

        H_APF:np.ndarray = np.matmul(
            np.matmul(R_MI_APF, self.target[i0:i3, j0:j3]),
            np.transpose(R_MI_APF),
        )  # Change Hessian

        if transpose is not None:
            H_APF = np.matmul(
                np.matmul(self.rot_X(np.pi), np.transpose(H_APF)),
                np.transpose(self.rot_X(np.pi)),
            )
            H_APF = np.matmul(
                np.matmul(self.rot_Z(np.pi), (H_APF)),
                np.transpose(self.rot_Z(np.pi)),
            )

        return list(H_APF.flatten()),Feature_AB,transpose