from __future__ import annotations

import faulthandler
import importlib
import os
import sys
from copy import deepcopy
from multiprocessing import Process, Queue
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from ase.units import Bohr
from scipy.spatial import distance_matrix

from hess_ml.src.decorator.decorator import checkTiming

if TYPE_CHECKING:
    from hess_ml.src.template import TrainMLHessianGFN2xTB

strings = [
        "response",
        "gap",
        "chem_pot",
        "HOAO_a",
        "LUAO_a",
        "HOAO_b",
        "LUAO_b",
        "delta_gap",
        "delta_chem_pot",
        "delta_HOAO",
        "delta_LUAO",
        "E_rep",
        "E_EHT",
        "E_disp2",
        "E_disp3",
        "E_ies_ixc",
        "E_AES",
        "E_AXC",
    ]

pattern = ""

for string in strings:
    pattern += string + "|"
pattern = pattern[:-1]

class FeatureTBlite:
    def __init__(self) -> None:
        return


    @checkTiming(enabled=False)
    def ImportFeature(self:TrainMLHessianGFN2xTB):
        #try:


        faulthandler.enable()
        from tblite.interface import Calculator

        charge = 0
        uhf = 0

        chargeFilePath = os.path.join(self.folder, ".CHRG")
        if os.path.isfile(chargeFilePath):
            with open(chargeFilePath) as chrg:
                charge = int(chrg.readline())

        uhfFilePath = os.path.join(self.folder, ".UHF")

        if os.path.isfile(uhfFilePath):
            with open(uhfFilePath) as uhfFile:
                uhf = int(uhfFile.readline())

        calc= Calculator(
        method="GFN2-xTB",
        uhf=uhf,
        charge=charge,
        numbers=self.NuclearCharge,
        positions=self.xyz*1/Bohr,
        )

        if self.solvent is not None:
            calc.add("alpb-solvation",self.solvent)

        calc.set("verbosity", 0)

        calc.add("bond-orders")

        calc.add("xtbml_xyz")
        res = calc.singlepoint()

        X:dict = deepcopy(res.get("post-processing-dict"))
        self.wbo = deepcopy(res.get("bond-orders"))
        self.gradient = np.array(deepcopy(res.get("gradient")))

        self.gradient /= Bohr

        X.pop("bond-orders")

        self.ml_feat = pd.DataFrame(deepcopy(X), columns=X.keys())

        X = None
        res = None
        calc = None

        self._get_dftd4_params()

        self.FilterFeatures()


        # except:

        #     self.do_calc = False
        #     calc = None
        #     res = None
        #     print("No convergenve structure will not be considered.")

    def comp_func(self,calc,result_queue):
        res = calc.singlepoint()
        result_queue.put(res)

    def ReadGradient(self:TrainMLHessianGFN2xTB, file):
        with open(file, "rb") as f:
            f.close()

        self.gradient = np.genfromtxt(
            file,
            skip_header=2 + self.N_atoms,
            skip_footer=1,
            loose=True,
        )

    def read_wbos(self:TrainMLHessianGFN2xTB,nats):
        wboFilePath = os.path.join(self.folder, "wbo")
        wbos = np.zeros([nats,nats])

        with open(wboFilePath) as file:
            lines = file.readlines()
            for line in lines:
                i,j,val = tuple(line.split())
                wbos[int(i)-1,int(j)-1] = float(val)
        file.close()

        wbos += wbos.T

        return wbos
    def _get_dftd4_params(self:TrainMLHessianGFN2xTB):

        dftd4_name = os.path.join(self.folder,"dftd4.out")

        os.system(f"dftd4 {self.xyz_file} > {dftd4_name}") # noqa: S605

        with open(os.path.join(self.folder,"dftd4.out")) as fname:
            lines = fname.readlines()
        fname.close()

        for j,line in enumerate(lines):
            if "     #    Z              CN          q   " in line:
                header_idx = j-2

            if "Molecular properties (in atomic units):" in line:
                footer_idx = len(lines)-j+2

        df = pd.read_csv(os.path.join(self.folder,"dftd4.out"),
                            names=["#","Z","CN","q","C6","C8"],
                            sep=r"\s+",header=header_idx,skipfooter=footer_idx,
                            engine="python")

        self.C6_params = df["C6"].to_numpy()

    def FilterFeatures(self:TrainMLHessianGFN2xTB):

        self.dipm = {}
        self.qm = {}
        self.q = {}
        self.cn = {}
        self.p = {}

        self.scalars = []
        self.vectors = []
        self.matrices = []

        self.scalar_keys = []
        self.vector_keys = []
        self.matrix_keys = []

        self.distance_mat = distance_matrix(self.xyz,self.xyz)

        vector = []
        matrix = []
        scalar = []

        for orb in ["s", "p", "d", "A", "e", "Z"]:

            if orb not in {"s", "p", "d"}:

                self.dipm[f"delta_{orb}"] = self.ml_feat.loc[:,
                    self.ml_feat.columns.str.contains(f"delta_dipm_{orb}_")].to_numpy()

                vector.append(self.dipm[f"delta_{orb}"])

                self.qm[f"delta_{orb}"] = self.ml_feat.loc[:,
                    self.ml_feat.columns.str.contains(f"delta_qm_{orb}_")].to_numpy()

                self.qm[f"delta_{orb}"] = self._transform_sym_mat_array(self.qm[f"delta_{orb}"])

                matrix.append(self.qm[f"delta_{orb}"])

                self.vector_keys.append(f"delta_dipm_{orb}")
                self.matrix_keys.append(f"delta_qm_{orb}")

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


        self.energy_based = self.ml_feat.loc[:,self.ml_feat.columns.str.contains(pattern)].to_numpy()
        self.scalar_keys.extend(strings)

        scalar.extend(self.energy_based.T)

        self.cn["default"] = self.ml_feat.loc[:, "CN"].to_numpy()
        self.cn["delta"] = self.ml_feat.loc[:, self.ml_feat.columns.str.contains("delta_CN")].to_numpy()

        self.scalar_keys.append("default_CN")
        self.scalar_keys.append("delta_CN")

        scalar.append(self.cn["default"])
        scalar.append(self.cn["delta"].flatten())

        self.q["default"] = self.ml_feat.loc[:, "q_A"].to_numpy()
        self.q["delta"] = self.ml_feat.loc[:, self.ml_feat.columns.str.contains("delta_q_A")].to_numpy()

        scalar.append(self.q["default"])

        scalar.append(self.q["delta"].flatten())

        self.scalar_keys.append("default_q_A")
        self.scalar_keys.append("delta_q_A")

        self.scalars, self.vectors ,self.matrices = self._transform_arrays(np.array(scalar),np.array(vector),np.array(matrix))


    def _transform_arrays(self:TrainMLHessianGFN2xTB,scalar:np.ndarray,vector:np.ndarray,matrix:np.ndarray):
        a,b = scalar.shape

        transformed_scalar = np.empty_like(scalar).reshape(b,a)
        a,b,c = vector.shape
        transformed_vector = np.empty_like(vector).reshape(b,a,c)
        a,b,c,d = matrix.shape
        transformed_matrix = np.empty_like(matrix).reshape(b,a,c,d)

        for i in range(self.N_atoms):
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
        return temp_mat + temp_mat.T - np.diag(np.diag(temp_mat))

