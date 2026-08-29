from __future__ import annotations
import os 

from typing import TYPE_CHECKING
from joblib import load

from mlhess.utils.io.reader import read_txt_file
from mlhess.utils.io.parser import parse_data_set
from mlhess.utils.chemistry.molecule import Molecule
from mlhess.machinelearning.feature.base_class import Feature
from mlhess.machinelearning.target.hessian import NuclearHessianPM, NuclearHessian
from mlhess.utils.patcher import patch_methods
from mlhess.calculator.tblite_wrapper import Calculator
from tcgm_lib.trv.trv_models.mrrho import TruhlarCramerRRHO
from tcgm_lib.IO.writer import wrt_hess_to_xtb

if TYPE_CHECKING:
    from mlhess.management.config import Configurator
    from sklearn.dummy import DummyRegressor


class PredictionHandling:
    def __init__(self, config: Configurator):
        """Initialize the handler for predictions.

        Args:
            config (Configurator): Configuration for the prediction run.
        """
        self.config = config
        self._model: DummyRegressor | None = None

    def collect_folders(self):
        """Parses data set folders with given information and adds a list of folders if given."""
        self.folder_names = parse_data_set(self.config)
        self.folder_names.extend(read_txt_file(self.config.collector.file_list))

        print(f"Total of {len(self.folder_names)} found.")

    def _pick_feature_class(self):
        """Pick a class to describe features.

        Returns:
            type[Feature]: Feature class used to represent molecular features.
        """
        return Feature

    def _pick_target_class(self):
        """Pick the target type for further calculations.

        Returns:
            AbstractTarget: Target class used for further calculations.
        """
        match self.config.molecule.target_class.lower():
            case "mlh":
                target_type = NuclearHessian
            case "mlh_pm":
                target_type = NuclearHessianPM

        return target_type

    @property
    def model(self) -> DummyRegressor:
        """ML model.

        Returns:
            DummyRegressor: The trained ML model.
        """
        if self._model is None:
            self._model = load(self.config.predict.model_name)
        return self._model

    @model.setter
    def model(self, model: DummyRegressor) -> None:
        """Set the ML model.

        Args:
            model (DummyRegressor): The ML model to use.
        """
        self._model = model

    def run_protocol(self):
        """Compute Hessians of collected structures and print thermodynamic properties."""

        self.collect_folders()

        methods = {
            "feature_class": self._pick_feature_class(),
            "hess_class": self._pick_target_class(),
        }

        with patch_methods(Molecule, methods):
            for fname in self.folder_names:
                mol = Molecule(fname, self.config.molecule.xyz_file)
                mol.access_ml_hessian = True

                mol.prepare_prediction(Calculator, self.config.general.threads)
                if mol.calc_succeeded:
                    mol.predict(self.model)
                    wrt_hess_to_xtb(os.path.join(mol.path,'MLhessian'),mol.ml_hessian.hessian)
                    mol.nuclear_properties.calc_properties(shift=False)
                    trv = TruhlarCramerRRHO(mol)
                    trv.print_thermoprop()
