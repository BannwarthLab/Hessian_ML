"""Package holding the Molecucle properties."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import tcgm_lib.convert.pse_converter as pse_cv
from tcgm_lib.molecule.molecule import Molecule as AbstractMolecule
from tcgm_lib.molecule.symmetry import Symmetry

from mlhess.utils.chemistry.nuclear_properties import NuclearProperties
from mlhess.utils.io.reader import read_xyz
from mlhess.calculator.tblite import Calculator

from mlhess.machinelearning.target.hessian import NuclearHessian
from mlhess.utils.chemistry.electronic_properties import ElectronicProperties
from mlhess.machinelearning.transform.transform_mlh import (
    transform_training,
    transform_prediction,
)
from mlhess.machinelearning.feature.base_class import Feature


if TYPE_CHECKING:
    from sklearn.dummy import DummyRegressor


class Molecule(AbstractMolecule):
    def __init__(  # noQA: PLR0913
        self,
        path: str,
        fxyz: str,
        symmetry: tuple[str, int] = ("c1", 1),
        charge: int | None = None,
        multiplicity: int | None = None,
        solvent: str | None = None,
    ) -> None:
        """Instantiate the a molecule with basic properties.

        Args:
            path (str): path to the folder.
            fxyz (str): filename of a xyz.
            elements (np.ndarray): array of elements. Defaults to np.array([], dtype=str).
            symmetry (tuple, optional): Symmetry of the molecule. Defaults to ("auto",1).
            charge (int | None, optional): charge. Defaults to None.
            multiplicity (int | None, optional): multiplicity. Defaults to None.
        """

        self.calc_succeeded = True

        self.path = path
        self.fxyz = fxyz
        elements, xyz = read_xyz(os.path.join(self.path, self.fxyz))
        self.xyz = xyz
        self.elements = [el.lower().capitalize() for el in elements]

        self._nat: None | int = None

        self.masses = pse_cv.elements_to_masses(self.elements)
        self.mol_mass = np.sum(self.masses)
        self.atomic_numbers = pse_cv.elements_to_atomic_numbers(self.elements)
        self.charge = charge
        self.multiplicity = multiplicity

        self.access_ml_hessian: bool = False

        self._solvent: str | None = solvent
        # self._feature: Feature | None = None
        self._feature: None = None
        self._feature_class = Feature
        self._trv_properties: None = None
        self._frequencies: np.ndarray | None = None
        self._gradient: np.ndarray | None = None
        self._processed_target: np.ndarray | None = None
        self.computed_atom_pairs: list = []

        self.symmetry = Symmetry(self, symmetry)
        self.nuclear_properties = NuclearProperties(self)
        self.electronic_properties = ElectronicProperties(self)

        self._hess_class = NuclearHessian
        self._hessian = None
        self._ml_hessian = None

        self._R_MI_APF_mat = np.zeros([self.nat * 3, self.nat * 3])
        self.transpose_list: list[list | None] = []

    @property
    def solvent(self) -> str | None:
        """Solvent."""
        return self._solvent

    @property
    def feature(self):
        """Features of the molecule.

        Returns:
            TBLiteFeatureCalc: Features computed with tblite.
        """
        if self._feature is None:
            self._feature: Feature = self._feature_class(self)
        return self._feature

    @property
    def processed_target(self):
        return self._processed_target

    @processed_target.setter
    def processed_target(self, vals: np.ndarray) -> None:
        self._processed_target = vals

    @property
    def feature_class(self):
        return self._feature_class

    @feature_class.setter
    def feature_class(self, feature_class: type[Feature]):
        self._feature_class = feature_class

    @property
    def hess_class(self):
        return self._hess_class

    @hess_class.setter
    def hess_class(self, hess_class: type[NuclearHessian]):
        self._hess_class = hess_class

    @property
    def hessian(self):
        """Hessian of the molecule.

        Returns:
            Hessian: Hessian
        """
        if self._hessian is None:
            self._hessian = self.hess_class(self)

        # Allows access with tcgm-lib
        if self.access_ml_hessian:
            return self._ml_hessian

        return self._hessian

    @property
    def ml_hessian(self):
        """Hessian of the molecule.

        Returns:
            Hessian: Hessian
        """
        if self._ml_hessian is None:
            self._ml_hessian = self.hess_class(self)
        return self._ml_hessian

    @property
    def nat(self) -> int:
        """Number of atoms.

        Returns:
            int: number of atoms derived from elements list length.
        """
        if self._nat is None:
            self._nat = len(self.elements)
        return self._nat

    def read_hessian(self, fhess, hesstype: str | None = "xTB") -> None:
        super().read_hessian(os.path.join(self.path, fhess), hesstype)
        self.hessian.apply_adaptation()
        return self.hessian.hessian

    @property
    def R_MI_APF_mat(self):
        return self._R_MI_APF_mat

    def prepare_training(self, calc_class: type[Calculator] = Calculator):
        if self._hessian is None:
            print("No hessian provided.")
            return
        calc = calc_class(self)
        calc.compute_feature()
        transform_training(self)
        return

    def prepare_prediction(
        self, calc_class: type[Calculator] = Calculator, num_threads=4
    ):
        calc = calc_class(self)
        calc.compute_feature()
        transform_prediction(self, num_threads)
        return

    def predict(self, model: DummyRegressor):
        self.target = model.predict(self.feature.processed)
        self.ml_hessian.get_processed_target()
        return self.ml_hessian.hessian
