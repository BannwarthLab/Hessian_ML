"""Setting up a class for configurations."""

from __future__ import annotations
import os 
import mlhess.utils.io.parser as parser
from mlhess.management.base_settings import PACKAGE_DIR

class GeneralConfig:
    def __init__(self) -> None:
        self.threads = 4
        self.gpu = True
        self.random_state = 79
        self.runtype = "mlhess"
        self.collector = True
        self.trainer = True
        self.predictor = False
        self.target = "hessian_pm"  # hessian


class CollectorConfig:
    def __init__(self):
        self.folder = None
        self.file_list = None
        self.gather = "molecules"


class MoleculeConfig:
    def __init__(self) -> None:
        self.feature_class = "xtbml"
        self.target_class = "mlh"
        self.xyz_file = "xtbopt.xyz"
        self.target_file = "hessian"
        self.solvent = None
        self.alphas = [1.0]  # TODO:Check if correct numbers [0.8,1.0,1.3]


class TrainConfig:
    def __init__(self) -> None:
        self.target_model = "mlh_l"  # mlhpm
        self.loss = "relHub"  # hub
        self.train_size = 0.75
        self.validation_size = 0.1
        self.test_size = 0.25
        self.method = "ETR"
        self.model_name = "MLH"
        self.parameter: list[dict] = [{}]
        self.error_parameter: list[dict] = [{}]

        self.active = False
        self.select = None
        self.scale = None
        self.transform = None


class InternalConfig:
    def __init__(self) -> None:
        self.train = None


class TrainParameterConfig:
    def __init__(self) -> None:
        pass


class PredictConfig:
    def __init__(self) -> None:
        self.model_name = os.path.join(PACKAGE_DIR,"/machinelearning/default_model/default_MLH.joblib")
        self.folder = None
        self.file_list = None


class Configurator:
    def __init__(self, input_file: str | None) -> None:
        if input_file is None:
            config = {}
        else:
            config = parser.parse_input_toml_file(input_file)

        self.molecule = MoleculeConfig()
        self.general = GeneralConfig()
        self.collector = CollectorConfig()
        self.train = TrainConfig()
        self.predict = PredictConfig()
        self.internal = InternalConfig()
        self._set_config(config)

    def _set_config(self, config: dict):
        for key in config:
            adapt_class = self._choose_config_class(key)
            for subkey in config[key]:
                name = subkey.lower()
                value = config[key][subkey]
                setattr(adapt_class, name, value)
            self._set_config_class(key, adapt_class)

    def _choose_config_class(self, key: str):
        class_name = key.lower().capitalize()
        if class_name == type(self.molecule).__name__[:-6]:
            adapt_class: (
                MoleculeConfig
                | CollectorConfig
                | GeneralConfig
                | TrainConfig
                | PredictConfig
            ) = self.molecule
        if class_name == type(self.collector).__name__[:-6]:
            adapt_class = self.collector
        if class_name == type(self.general).__name__[:-6]:
            adapt_class = self.general
        if class_name == type(self.train).__name__[:-6]:
            adapt_class = self.train
        if class_name == type(self.predict).__name__[:-6]:
            adapt_class = self.predict

        return adapt_class

    def _set_config_class(self, key: str, adapt_class):
        class_name = key.lower().capitalize()
        if class_name == type(self.molecule).__name__[:-6]:
            self.molecule = adapt_class
        if class_name == type(self.collector).__name__[:-6]:
            self.collector = self.collector
        if class_name == type(self.general).__name__[:-6]:
            self.general = adapt_class
        if class_name == type(self.train).__name__[:-6]:
            self.train = adapt_class
        if class_name == type(self.predict).__name__[:-6]:
            self.predict = adapt_class

        return adapt_class
