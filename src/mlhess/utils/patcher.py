from __future__ import annotations
from typing import TYPE_CHECKING
from contextlib import contextmanager
from mlhess.machinelearning.feature.base_class import Feature
from mlhess.machinelearning.target.hessian import NuclearHessian, NuclearHessianPM
from mlhess.utils.chemistry.molecule import Molecule


if TYPE_CHECKING:
    from mlhess.management.config import Configurator


@contextmanager
def patch_methods(cls, methods: dict):
    # Save originals
    originals = {name: getattr(cls, name) for name in methods}
    # Apply patches
    for name, func in methods.items():
        setattr(cls, name, func)
    try:
        yield
    finally:
        # Restore originals
        for name, func in originals.items():
            setattr(cls, name, func)


def patch_molecule(config):
    methods = {
        "feature_class": pick_feature_class(config),
        "hess_class": pick_target_class(config),
    }
    patch_methods(Molecule, methods)


def pick_feature_class():
    """pick a class to describe features.

    :return: _description_
    :rtype: _type_
    """
    return Feature


def pick_target_class(config: Configurator):
    """Pick the target type for further calculations.

    :return: target class
    :rtype: AbstractTarget
    """
    match config.molecule.target_class.lower():
        case "mlh":
            target_type = NuclearHessian
        case "mlh_pm":
            target_type = NuclearHessianPM

    return target_type
