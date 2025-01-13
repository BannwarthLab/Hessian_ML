"""Setting up a class for configurations."""
from __future__ import annotations


class GeneralConfig:
    def __init__(self) -> None:
        self.runtype = "hessian"
        self.feature = True
        self.train = True
        self.predict = False
        self.optimization = False
        self.random_state = 79
        self.threads = 1
        self.wrt_feature = True
        self.split_feature = True

class MoleculeConfig:
    def __init__(self) -> None:
        self.folder = None
        self.files = None
        self.feature = "tblite"
        self.target_file = "hessian"
        self.target_type = "vanilla"
        self.xyz_file = "xtbopt.xyz"
        self.program = ["xtb"]
        self.hessian_origin = "xtb"
        self.feature_file = None
        self.solvent = None

class TrainConfig:
    def __init__(self) -> None:
        self.train_size = 0.75
        self.test_size = 0.25
        self.method = "ETR"
        self.model_name = "ML_Hess"
        self.parameter = [{}]
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
        self.model_name = "ML_Hess"
        self.folder = None
        self.folder_list= None

class Configurations:
    def __init__(self,config) -> None:
        self.molecule = MoleculeConfig()
        self.general  = GeneralConfig()
        self.train = TrainConfig()
        self.predict = PredictConfig()
        self.interal = InternalConfig()
        self._set_config(config)

    def _set_config(self,config:dict):

        for key in config:

            adapt_class = self._choose_config_class(key)
            for subkey in config[key]:
                name = subkey.lower()
                value = config[key][subkey]
                setattr(adapt_class,name,value)
            self._set_config_class(key,adapt_class)


    def _choose_config_class(self,key:str):

        class_name = key.lower().capitalize()
        if class_name == type(self.molecule).__name__[:-6]:
            adapt_class = self.molecule
        if class_name == type(self.general).__name__[:-6]:
            adapt_class = self.general
        if class_name == type(self.train).__name__[:-6]:
            adapt_class = self.train
        if class_name == type(self.predict).__name__[:-6]:
            adapt_class = self.predict

        return adapt_class


    def _set_config_class(self,key:str,adapt_class):

        class_name = key.lower().capitalize()
        if class_name == type(self.molecule).__name__[:-6]:
            self.molecule = adapt_class
        if class_name == type(self.general).__name__[:-6]:
            self.general= adapt_class
        if class_name == type(self.train).__name__[:-6]:
            self.train = adapt_class
        if class_name == type(self.predict).__name__[:-6]:
            self.predict = adapt_class

        return adapt_class
