"Put the tblite calculation frame work here"

from __future__ import annotations

import faulthandler
import os
import subprocess
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd


from scipy.spatial import distance_matrix

from scipy.constants import physical_constants
from copy import deepcopy
from mlhess.utils.io.parser import parse_dftd4_output
from dftd4.interface import DispersionModel
from tblite.interface import Calculator as TBCalculator

bohr_in_m, _, _ = physical_constants["Bohr radius"]
Bohr = bohr_in_m * 1e10

if TYPE_CHECKING:
    from mlhess.utils.chemistry.molecule import Molecule

strings = [
    "response",
    # "gap",
    # "chem_pot",
    # "HOAO_a",
    # "LUAO_a",
    # "HOAO_b",
    # "LUAO_b",
    # "delta_gap",
    # "delta_chem_pot",
    # "delta_HOAO",
    # "delta_LUAO",
    "E_rep",
    "E_eht",
    "E_disp2",
    "E_disp3",
    "E_ies_ixc",
    "E_aes",
    "E_axc",
    "E_tot",
]

pattern = ""

for string in strings:
    pattern += string + "|"
pattern = pattern[:-1]


strings = [
    "response_alpha",
    # "gap",
    # "chem_pot",
    # "HOAO_a",
    # "LUAO_a",
    # "HOAO_b",
    # "LUAO_b",
    # "delta_gap",
    # "delta_chem_pot",
    # "delta_HOAO",
    # "delta_LUAO",
    "E_rep",
    "E_eht",
    "E_disp2",
    "E_disp3",
    "E_ies_ixc",
    "E_aes",
    "E_axc",
    "E_tot",
]

pattern_uhf = ""

for string in strings:
    pattern_uhf += string + "|"
pattern_uhf = pattern_uhf[:-1]


class Calculator:
    def __init__(self, mol: Molecule) -> None:
        self._mol = mol
        # self._processed_features: np.ndarray | list | None = None
        self.new_keys: None | list = None
        return

    # @property
    # def processed_features(self) -> np.ndarray:
    #     if self._processed_features is None:
    #         self.get_processed_features()
    #     return np.array(self._processed_features)

    # def get_processed_features(self):
    #     self.compute_feature()
    #     self._processed_features = self.scalars.flatten().tolist()
    #     self._processed_features.extend(self.vectors.flatten().tolist())
    #     self._processed_features.extend(self.matrices.flatten().tolist())

    def compute_feature(self):

        try:
            faulthandler.enable()
            os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libgomp.so.1"

            calc = TBCalculator(
                method="GFN2-xTB",
                uhf=self._mol.electronic_properties.uhf,
                charge=self._mol.electronic_properties.charge,
                numbers=np.array(self._mol.atomic_numbers),
                positions=self._mol.xyz * 1 / Bohr,
            )

            if self._mol.solvent is not None:
                calc.add("alpb-solvation", self._mol.solvent)

            calc.set("verbosity", 0)

            calc.add("bond-orders")

            calc.add("xtbml.toml")

            res = calc.singlepoint()

            X: dict = deepcopy(res.get("post-processing-dict"))

            self.wbo = deepcopy(res.get("bond-orders"))

            self.energy = deepcopy(res.get("energy"))

            self.gradient = np.array(deepcopy(res.get("gradient")))

            self.gradient /= Bohr

            X.pop("bond-orders")

            new_keys = self.adapt_keys(X.keys())

            self.ml_feat = pd.DataFrame(deepcopy(X), columns=X.keys())

            self.ml_feat = self.ml_feat.rename(columns=new_keys)

            X = None
            res = None
            calc = None

            self._get_dftd4_params()

            self._filter_features()

        except:  # noqa: E722
            self._mol.calc_succeeded = False
            calc = None
            res = None
            print("No convergence structure will not be considered.")

    def adapt_keys(self, keys):
        new_keys = {}
        for old_key in keys:
            if "." in old_key:
                new_key = old_key[: old_key.rindex("_")]
            else:
                new_key = old_key

            new_keys[old_key] = new_key

        return new_keys

    def read_wbos(self):
        wboFilePath = os.path.join(self._mol.path, "wbo")
        wbos = np.zeros([self._mol.nat, self._mol.nat])

        with open(wboFilePath) as file:
            lines = file.readlines()
            for line in lines:
                i, j, val = tuple(line.split())
                wbos[int(i) - 1, int(j) - 1] = float(val)
        file.close()

        wbos += wbos.T

        return wbos

    def _get_dftd4_params_old(self):
        fxyz = os.path.join(self._mol.path, self._mol.fxyz)
        result = subprocess.run(
            ["dftd4", fxyz], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        self._mol.feature.C6params = parse_dftd4_output(result.stdout)

    def _get_dftd4_params(self):
        disp = DispersionModel(
            positions=self._mol.xyz * 1 / Bohr,
            numbers=np.array(self._mol.atomic_numbers),
        )
        props = disp.get_properties()
        self._mol.feature.C6params = np.float32(np.diag(props["c6 coefficients"]))

    def _filter_features(self):
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

        self._mol.feature.distance_mat = distance_matrix(self._mol.xyz, self._mol.xyz)

        vector = []
        matrix = []
        scalar = []

        for orb in ["s", "p", "d", "A"]:  # , "e", "Z"]:
            if orb not in {"s", "p", "d"}:
                self.dipm[f"delta_{orb}"] = self.ml_feat.loc[
                    :, self.ml_feat.columns.str.contains(f"ext_dipm_{orb}_")
                ].to_numpy()

                for idx in range(0, (self.dipm[f"delta_{orb}"]).shape[1], 3):
                    vector.append(self.dipm[f"delta_{orb}"][:, idx : idx + 3])

                # self.qm[f"delta_{orb}"] = self.ml_feat.loc[:,
                #     self.ml_feat.columns.str.contains(f"ext_qm_{orb}_")].to_numpy()

                # self.qm[f"delta_{orb}"] = self._transform_sym_mat_array(self.qm[f"delta_{orb}"])

                # matrix.append(self.qm[f"delta_{orb}"])

                self.matrix_keys.append(f"delta_qm_{orb}")
                # self.vector_keys.append(f"delta_dipm_{orb}")

                for key in [f"ext_dipm_{orb}"]:  # ,f"delta_qm_{orb}"
                    temp = self.ml_feat.loc[:, key].to_numpy()
                    scalar.extend(temp.reshape(-1, self._mol.nat))
                    self.scalar_keys.append("delta" + key[3:])
                    self.norms["delta" + key[3:]] = temp

            if orb not in {"e", "Z"}:
                self.dipm[f"{orb}"] = self.ml_feat.loc[
                    :, self.ml_feat.columns.str.startswith(f"dipm_{orb}_")
                ].to_numpy()

                self.qm[f"{orb}"] = self.ml_feat.loc[
                    :, self.ml_feat.columns.str.startswith(f"qm_{orb}_")
                ].to_numpy()

                self.qm[f"{orb}"] = self._transform_sym_mat_array(self.qm[f"{orb}"])

                vector.append(self.dipm[f"{orb}"])
                matrix.append(self.qm[f"{orb}"])

                self.vector_keys.append(f"dipm_{orb}")
                self.matrix_keys.append(f"qm_{orb}")

                if orb != "A":
                    self.p[f"{orb}"] = self.ml_feat.loc[
                        :, self.ml_feat.columns.str.contains(f"p_{orb}")
                    ].to_numpy()
                    scalar.append(self.p[f"{orb}"].flatten())
                    self.scalar_keys.append(f"p_{orb}")

                for key in [f"dipm_{orb}", f"qm_{orb}"]:
                    temp = self.ml_feat.loc[:, key].to_numpy()
                    scalar.extend(temp.reshape(-1, self._mol.nat))
                    self.scalar_keys.append("delta" + key[3:])
                    self.norms["delta" + key[3:]] = temp

        self.dipm_norm = np.linalg.norm(self.dipm["A"], axis=1)

        if "response_alpha" in self.ml_feat.keys():
            self.energy_based = self.ml_feat.loc[
                :, self.ml_feat.columns.str.contains(pattern_uhf)
            ].to_numpy()
        else:
            self.energy_based = self.ml_feat.loc[
                :, self.ml_feat.columns.str.contains(pattern)
            ].to_numpy()

        self.scalar_keys.extend(strings)

        scalar.extend(self.energy_based.T)

        self.cn["default"] = self.ml_feat.loc[:, "CN_A"].to_numpy()
        self.cn["delta"] = self.ml_feat.loc[
            :, self.ml_feat.columns.str.startswith("ext_CN")
        ].to_numpy()

        self.scalar_keys.append("default_CN")
        self.scalar_keys.append("delta_CN")

        scalar.append(self.cn["default"])
        scalar.extend(self.cn["delta"].reshape(-1, self._mol.nat))

        self.q["default"] = self.ml_feat.loc[:, "q_A"].to_numpy()
        self.q["delta"] = self.ml_feat.loc[
            :, self.ml_feat.columns.str.contains("ext_q_A")
        ].to_numpy()

        scalar.append(self.q["default"])

        scalar.extend(self.q["delta"].reshape(-1, self._mol.nat))

        self._mol.feature.q = self.q
        self._mol.feature.dipm = self.dipm
        self._mol.feature.qm = self.qm
        self._mol.feature.dipm_norm = self.dipm_norm

        self.scalar_keys.append("default_q_A")
        self.scalar_keys.append("delta_q_A")

        self._mol.feature.scalar_keys = self.scalar_keys

        self._mol.scalar_keys = self.scalar_keys

        self._mol.feature.wbo = self.wbo
        (
            self._mol.feature.scalars,
            self._mol.feature.vectors,
            self._mol.feature.matrices,
        ) = self._transform_arrays(np.array(scalar), np.array(vector), np.array(matrix))

    def _transform_arrays(
        self, scalar: np.ndarray, vector: np.ndarray, matrix: np.ndarray
    ):
        a, b = scalar.shape

        transformed_scalar = np.empty_like(scalar).reshape(b, a)
        a, b, c = vector.shape
        transformed_vector = np.empty_like(vector).reshape(b, a, c)
        a, b, c, d = matrix.shape
        transformed_matrix = np.empty_like(matrix).reshape(b, a, c, d)

        for i in range(self._mol.nat):
            transformed_scalar[i, :] = scalar[:, i]
            transformed_vector[i, :, :] = vector[:, i, :]
            transformed_matrix[i, :, :, :] = matrix[:, i, :, :]

        return transformed_scalar, transformed_vector, transformed_matrix

    def _transform_sym_mat_array(self, mat_array):
        mat_array_new = []
        for atom in range(len(mat_array)):
            temp_mat = np.zeros([3, 3])
            temp_mat[np.tril_indices(temp_mat.shape[0], k=0)] = np.array(
                mat_array[atom]
            )
            temp_mat = temp_mat + temp_mat.T - np.diag(np.diag(temp_mat))
            mat_array_new.append(temp_mat)

        return np.array(mat_array_new)

    def _transform_sym_mat(self, mat):
        temp_mat = np.zeros([3, 3])
        temp_mat[np.tril_indices(temp_mat.shape[0], k=0)] = np.array(mat)
        temp_mat = temp_mat + temp_mat.T - np.diag(np.diag(temp_mat))

        return temp_mat
