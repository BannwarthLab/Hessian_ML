"""Holds geometric properties of molecule."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mlhess.utils.chemistry.molecule import Molecule


class NuclearProperties:
    """Holds geometric properties of molecule."""

    def __init__(self, molecule: Molecule) -> None:
        """Initialize the NuclearProperties object.

        Args:
            molecule (Molecule): Molecule object.
        """
        self.molecule = molecule

    def calc_properties(self, shift: bool) -> None:
        """Build the properties of a molecule.

        Args:
            shift (bool): If True, keep coordinates shifted to the center of
                mass. If False, shift them back to their original position.
        """
        self.center_of_mass = self.calc_center_of_mass()

        self._shiftCOM()

        self.ieval, self.ievec = self.calc_it()
        # self.calc_it_full_prec()

        self.avmom, self.islin, self.Ra, self.Rb, self.Rc = self.check_lin()

        # Shift xyz back.
        if not shift:
            self.molecule.xyz += self.center_of_mass
            self.shifted = False

    def calc_center_of_mass(self) -> np.ndarray:
        """Computes center of mass (COM) from cartesian coordinates.

        .. math::
            \\mathbf{s}  = \\frac{1}{M} \\sum_i^N m_i \\mathbf{x_i}

        Returns:
            np.ndarray: Center of mass, dim: 3.
        """
        ws = np.sum(self.molecule.xyz * self.molecule.masses[:, np.newaxis], axis=0)
        return ws / self.molecule.mol_mass

    def print_center_of_mass(self) -> None:
        """Prints the center of mass of the molecule."""
        print("Center of mass is at:")
        print("x: ", self.center_of_mass[0], "\n")
        print("y: ", self.center_of_mass[1], "\n")
        print("z: ", self.center_of_mass[2], "\n")

    def _shiftCOM(self) -> None:
        """Apply shift to cartesian coordinates."""
        self.molecule.xyz -= self.center_of_mass
        self.shifted = True

    def calc_it(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculates the Inertia Tensor.

        .. math::
           \\mathbf{I} = \\sum_i^N m_i  \\begin{pmatrix}
                 y_i^2+z_i^2 & -y_i x_i & - z_i x_i \\\\
                 -x_i y_i & x_i^2+z_i^2 &- z_i y_i \\\\
                -x_i z_i & - y_i z_i & x_i^2+y_i^2 \\\\
            \\end{pmatrix}

        Returns:
            tuple[np.ndarray, np.ndarray]: Eigenvalues and eigenvectors of the
                inertia tensor (in atomic mass unit * Bohr^2).
        """
        self.itens = np.zeros((3, 3), dtype=float)

        x, y, z = (
            self.molecule.xyz.T
        )  # Transpose for efficient access to x, y, and z arrays

        self.itens[0, 0] = np.sum(self.molecule.masses * (y**2 + z**2))  # y**2 + z**2
        self.itens[1, 0] = -np.sum(self.molecule.masses * (x * y))  # -y*x
        self.itens[2, 0] = -np.sum(self.molecule.masses * (x * z))  # -z*x
        self.itens[1, 1] = np.sum(self.molecule.masses * (z**2 + x**2))  # z**2 + x**2
        self.itens[2, 1] = -np.sum(self.molecule.masses * (y * z))  # -z*y
        self.itens[2, 2] = np.sum(self.molecule.masses * (x**2 + y**2))  # x**2 + y**2

        # helpers.print_matrix(self.itens)
        # Diagonalize inertia tensor
        return np.linalg.eigh(self.itens)

    def check_lin(self, thres: float = 3e-4) -> tuple[float, bool, float, float, float]:
        """Checks if a molecule is linear based on eigenvalues of its inertia tensor.

        Args:
            thres (float, optional): Threshold for linearity check. Defaults to 3e-4 (amu*angstrom^2).

        Returns:
            float: Average momentum (kg*m/s).
            bool: True if the molecule is linear, False otherwise.
            float: Rotational constant aa (cm^-1).
            float: Rotational constant bb (cm^-1).
            float: Rotational constant cc (cm^-1).
        """
        # print(f"{evals}")
        conv2 = 16.8576522
        amuang2_kgm2 = 1.66053  ## * 10 e -47
        mHz_rcm = 2.9979245e4

        islin: bool = False
        avmom = np.sum(self.ieval) / 2.0

        rot = np.zeros(3)
        # linearity check: look for prolate top: a = 0 != b = c
        if np.abs(self.ieval[1] - avmom + self.ieval[2] - avmom) <= thres:
            print("Molecule is linear!")
            islin = True
        self.ieval[self.ieval <= thres] = 0.0

        mask = self.ieval >= thres / 3

        rot = np.zeros(len(self.ieval))
        rot[mask] = mHz_rcm * conv2 / self.ieval[mask]

        xyzmom = self.ieval * amuang2_kgm2

        cc, bb, aa = rot / mHz_rcm

        avmom = 1e-47 * (np.sum(xyzmom) / len(xyzmom))

        return avmom, islin, aa, bb, cc
