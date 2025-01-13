"""Package holding the Molecucle properties."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import tcgm_lib.convert.pse_converter as pse_cv
from tcgm_lib.molecule.molecule import Molecule as AbstractMolecule
from tcgm_lib.molecule.nuclear_hessian import NuclearHessian
from tcgm_lib.molecule.nuclear_properties import NuclearProperties
from tcgm_lib.IO.readers import read_xyz
#from tcgm_lib.molecule.symmetry import Symmetry

from hess_ml.src2.molecule.electronic_properties import ElectronicProperties
from hess_ml.src2.molecule.tblite.prediction_feature import Feature

if TYPE_CHECKING:
    from tcgm_lib.trv.trv_models.rrho import RRHO

class Molecule(AbstractMolecule):

    def __init__(  # noQA: PLR0913
        self,
        path:str,
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
        elements, xyz  = read_xyz(os.path.join(self.path,self.fxyz))
        self.xyz = xyz
        self.elements = [el.lower().capitalize() for el in elements]
        
        self._nat: None | int = None

        self.masses = pse_cv.elements_to_masses(self.elements)
        self.mol_mass = np.sum(self.masses)
        self.atomic_numbers = pse_cv.elements_to_atomic_numbers(self.elements)
        self.charge = charge
        self.multiplicity = multiplicity

        self._solvent: str | None = solvent
        self._feature: Feature | None = None

        self._trv_properties: RRHO | None = None
        self._frequencies: np.ndarray | None = None
        self._gradient: np.ndarray | None = None

        
        self.symmetry = None#Symmetry(self, symmetry)
        self.nuclear_properties = NuclearProperties(self)
        self.electronic_properties = ElectronicProperties(self)

        self.hessian = NuclearHessian(self)
        self.ml_hessian = NuclearHessian(self)

    @property
    def solvent(self) -> str|None:
        """Solvent."""
        return self._solvent

    @property
    def feature(self):
        """Features of the molecule.

        Returns:
            TBLiteFeatureCalc: Features computed with tblite.
        """
        if self._feature is None:
            self._feature = Feature(self)
        return self._feature

    @feature.setter
    def feature(self, feature: type[Feature]) -> None:
        self._feature = feature(self)

    @property
    def nat(self) -> int:
        """Number of atoms.

        Returns:
            int: number of atoms derived from elements list length.
        """
        if self._nat is None:
            self._nat = len(self.elements)
        return self._nat

    def read_hessian(self,fhess, hesstype: str | None = "xTB") -> None:
        return super().read_hessian(os.path.join(self.path,fhess), hesstype)
