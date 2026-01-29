"""Electronic properties of a molecule."""
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hess_ml.src2.molecule.molecule import Molecule


class ElectronicProperties:

    def __init__(self,mol) -> None:
        self._mol = mol
        self._atomic_dipole: np.ndarray | None = None
        self._uhf = None
        self._charge = None

    @property
    def charge(self):
        """Charge of the moleucle.

        If the charge is not set, in the path of the Molecule object is searched for a '.CHRG' file.
        If it is not found the charge is set to zero.
        """
        if self._charge is None:
            chargeFilePath = os.path.join(self._mol.path, ".CHRG")
            if os.path.isfile(chargeFilePath):
                with open(chargeFilePath) as chrg:
                    self._charge = int(chrg.readline())
            else:
                self._charge = 0
        return self._charge

    @charge.setter
    def charge(self, charge: int) -> None:
        """Charge.
        Args:
            charge (np.ndarray): charge
        """
        self._charge = charge

    @property
    def uhf(self) -> int:
        """Multiplicity of the moleucle.

        If the multiplicity is not set, in the path of the Molecule object is searched for a '.UHF' file.
        If it is not found the multiplicity is set to zero.
        """
        if self._uhf is None:
            uhfFilePath = os.path.join(self._mol.path, ".UHF")
            if os.path.isfile(uhfFilePath):
                with open(uhfFilePath) as uhfFile:
                    self._uhf = int(uhfFile.readline())
            else:
                self._uhf = 0
        return self._uhf

    @uhf.setter
    def uhf(self, uhf: int) -> None:
        """Multiplicity.
        Args:
            uhf (np.ndarray): multiplicity
        """
        self._uhf = uhf

    @property
    def atomic_dipole(self):
        """Atomic dipole moments"""
        return self._atomic_dipole

    @atomic_dipole.setter
    def atomic_dipole(self, atomic_dipole:np.ndarray) -> None:
        """Set the atomic dipole moments.

        Args:
            atomic dipole moments (np.ndarray): atomic dipole moments
        """
        assert len(atomic_dipole) == self._mol.nat, "Dimension of gradient does not correspond to number of atoms."
        self.atomic_dipole = atomic_dipole
