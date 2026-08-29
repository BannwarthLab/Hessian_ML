"""Transforms the constants of the PSE from one to another."""

from __future__ import annotations

import numpy as np
from tcgm_lib.constants.periodic_table import (
    ATOMIC_NUMBERS_TO_ELEMENTS,
    ATOMIC_NUMBERS_TO_MASSES,
    ELEMENTS_TO_ATOMIC_NUMBERS,
    ELEMENTS_TO_MASSES,
    MASSES_TO_ATOMIC_NUMBERS,
    MASSES_TO_ELEMENTS,
)


def elements_to_atomic_numbers(elements: np.ndarray | list) -> list:
    """Transform elements to atomic number.

    Args:
        elements (np.ndarray | list): array of elements

    Returns:
        list: array of atomic numbers
    """
    return [ELEMENTS_TO_ATOMIC_NUMBERS[el] for el in elements]


def elements_to_masses(elements: np.ndarray | list) -> np.ndarray:
    """Transform elements to masses.

    Args:
        elements (np.ndarray | list): array of elements

    Returns:
        np.ndarray: array of masses
    """
    return np.array([ELEMENTS_TO_MASSES[el] for el in elements])


def atomic_numbers_to_elements(atomic_numbers: np.ndarray | list) -> list:
    """Transform atomic number to atomic elements.

    Args:
        atomic_numbers (np.ndarray | list): array of atomic number

    Returns:
        list: array of elements
    """
    return [ATOMIC_NUMBERS_TO_ELEMENTS[at] for at in atomic_numbers]


def atomic_numbers_to_masses(atomic_numbers: np.ndarray | list) -> np.ndarray:
    """Transform atomic number to masses.

    Args:
        atomic_numbers (np.ndarray | list): array of atomic number

    Returns:
        np.ndarray: array of masses
    """
    return np.array([ATOMIC_NUMBERS_TO_MASSES[at] for at in atomic_numbers])


def masses_to_elements(masses: np.ndarray | list) -> list:
    """Transform masses to elements.

    Args:
        masses (np.ndarray | list): array of masses

    Returns:
        list: array of elements
    """
    return [MASSES_TO_ELEMENTS[m] for m in masses]


def masses_to_atomic_numbers(masses: np.ndarray | list) -> list:
    """Transform masses to atomic number.

    Args:
        masses (np.ndarray | list): array of masses

    Returns:
        list: array of atomic numbers
    """
    return [MASSES_TO_ATOMIC_NUMBERS[m] for m in masses]
