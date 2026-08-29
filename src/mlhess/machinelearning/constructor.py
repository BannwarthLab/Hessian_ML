from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import torch
from joblib import dump,load # type: ignore
# Model pipeline
from sklearn.pipeline import Pipeline # type: ignore

# Feature scaling 
from sklearn.preprocessing import Normalizer, StandardScaler # type: ignore

# Feature Selection
from sklearn.decomposition import PCA, TruncatedSVD # type: ignore
from sklearn.feature_selection import VarianceThreshold # type: ignore

# ML Model
from mlhess.machinelearning.architecture.neural_nets import MLH_l, MLH_s

# loss functions
from mlhess.machinelearning.loss.huber import RelHuberLoss

# optimizer
from mlhess.machinelearning.optimizer import PyTorchRegressor
from mlhess.management.base_settings import PACKAGE_DIR

if TYPE_CHECKING:
    from mlhess.management.config import TrainConfig


class Constructor:
    def __init__(self, train_config: TrainConfig):
        self.config = train_config
        self.model_info: list[tuple] = []

    def build_model(self):
        self.set_scaler(self.config.scale)
        self.set_selection(self.config.select)
        self.set_model(self.config.method)
        self.set_pipeline()

    def set_model(self, method: str) -> list:
        """
        Define the ML model.
        """
        self.method = method

        params = self.config.parameter[0]

        method = method.lower()

        if method.lower() in ["mlh_l", "mlh_s"]:
            lr, gamma, epochs, batch_size = self.get_nn_params(params)
            model = self._choose_model(method)
            criterion = self._choose_loss(
                self.config.loss, self.config.error_parameter[0]
            )
            regr_model = PyTorchRegressor(
                lr, gamma, epochs, model, criterion, batch_size
            )

        else:
            print("Model does not match to the existing models.")
            sys.exit()

        self.model_info.append(("regressor", regr_model))

        return self.model_info

    def _choose_model(self, method: str):
        if method.lower() == "mlh_l":
            model:type[MLH_l]|type[MLH_s] = MLH_l
        elif method.lower() == "mlh_s":
            model = MLH_s

        return model

    def _choose_loss(self, name: str, error_parameter: dict) -> torch.nn.HuberLoss|RelHuberLoss:
        if name.lower() in ["huber_loss", "huberloss"]:
            return torch.nn.HuberLoss(delta=error_parameter.get("delta", 5e-2))
        elif name.lower() in ["relhuber", "rel_huber", "relhuber_loss", "relhuberloss"]:
            return RelHuberLoss(
                    delta=error_parameter.get("delta", 5e-3),
                    relative_delta=error_parameter.get("rel_delta", 1e-4))
        else:
            print("Loss name not found. Falling back to HuberLoss.")
            return torch.nn.HuberLoss(delta=error_parameter.get("delta", 5e-2))
        
    def set_scaler(self, scaling: str | None = None) -> None:
        """
        Define the standard scaler for the ML model.
        """

        if scaling is None:
            return

        print("Setting scaler")

        if scaling.lower() == "standardscaler":
            scaler = StandardScaler()
            self.model_info.append(("scaler", scaler))

        if scaling.lower() in ["normalizer"]:
            scaler = Normalizer()
            self.model_info.append(("scaler", scaler))
        return

    def set_selection(self, selection: str | None = None) -> None:
        """Define the feature selector for the ML model.

        The selected feature selector is appended to ``self.model_info``,
        the list of pipeline steps used to build the model.

        Args:
            selection (str, optional): Name of the feature selection method
                to use. Defaults to None.
        """
        if selection is None:
            return

        print("")
        print("Setting selection")

        if selection.lower() in ["svd", "truncatedsvd"]:
            feature_selector = TruncatedSVD(n_components=300)
            self.model_info.append(("feature_selector", feature_selector))

        elif selection.lower() in ["variancethreshold"]:
            feature_selector = VarianceThreshold()
            self.model_info.append(("feature_selector", feature_selector))

        elif selection.lower() in ["pca"]:
            feature_selector = PCA(n_components=0.999, svd_solver="auto")
            self.model_info.append(("feature_selector", feature_selector))

        else:
            print("Selection Method was not recognized. Choose a different method.")

        return

    def get_nn_params(self, params: dict):
        lr = params.get("learning_rate", 0.001)
        gamma = params.get("gamma", 0.99)
        epochs = params.get("epochs", 90)
        batch_size = params.get("batch_size", 1024)
        return lr, gamma, epochs, batch_size

    def set_pipeline(self):
        """
        Define the complete pipeline
        """

        self.complete_model = Pipeline(steps=self.model_info)

    def set_rnd_seed(self, seed):
        self.rnd_seed = seed

    def set_split(self, split=None):
        self.i_split = split

    def dump_model(self):
        model_name = self.config.model_name

        if self.i_split is not None:
            os.mkdir(f"Model{self.i_split}_{self.rnd_seed}")
            pathname = f"Model{self.i_split}_{self.rnd_seed}/{model_name}.joblib"
        else:
            pathname = f"{model_name}.joblib"

        dump(self.complete_model, pathname)

        print(f"Model is saved in {pathname}.\n")

    def training(self, features, targets, shuffle_idx):
        self.print_params()
        self.complete_model.fit(
            features[shuffle_idx],
            targets[shuffle_idx],
        )
        self.score = self.complete_model.score(
            features[shuffle_idx], targets[shuffle_idx]
        )
        print(
            f"Score on training data: {self.score}",
        )
        self.dump_model()

    def print_params(self):
        print("Parameters for the Model:")
        param_temp = self.complete_model.get_params()
        print(param_temp)
        for param in param_temp:
            print(f"{param}: {param_temp[param]}")


def load_model(path=os.path.join(PACKAGE_DIR,"machinelearning/default_model/default_MLH.joblib")):
    return load(path)