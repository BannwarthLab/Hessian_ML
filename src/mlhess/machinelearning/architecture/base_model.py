from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from mlhess.machinelearning.architecture.neural_nets import MLH_l
from mlhess.machinelearning.optimizer import PyTorchRegressor
from mlhess.machinelearning.loss.huber import RelHuberLoss
from sklearn.metrics import r2_score

if TYPE_CHECKING:
    from mlhess.management.config import Configurator


class DummyTransformer:
    def __init__(self):
        pass

        self.is_fitted_ = True

    def fit(*args, **kwargs):
        pass

    def transform(self, feature, *args, **kwargs):
        return feature

    def fit_transform(self, feature, *args, **kwargs):
        return feature


class MachineLearningModel:
    def __init__(self, config: Configurator):
        self.model = MLH_l
        self.selector = DummyTransformer()
        self.scaler = DummyTransformer()

        self.fit_history: list[str] = []
        self.config = config

    def __call__(self, *args, **kwds):
        self.predict(*args)

    def predict(self, feature):
        feature = self.scaler.transform(feature)
        feature = self.selector.transform(feature)

        if hasattr(self, "is_fitted_"):
            if not self.is_fitted_:
                raise ValueError("The model instance is not fitted yet.")
        else:
            self.is_fitted_ = True

        self.model.eval()
        self.model.to("cpu")

        with torch.no_grad():
            X_tensor = torch.from_numpy(feature).to("cpu")
            predictions: torch.Tensor = self.model(X_tensor)

        return predictions.cpu().numpy()

    def prep_fit_transform(self, feature):
        feature = self.scaler.fit_transform(feature)
        feature = self.selector.fit_transform(feature)
        return feature

    def prep_transform(self, feature):
        feature = self.scaler.transform(feature)
        feature = self.selector.transform(feature)
        return feature

    def fit(self, feature, target):
        feature = self.prep_fit_transform(feature)
        lr, gamma, epochs, batch_size = self._get_nn_params()
        criterion = self._choose_loss()
        regr = PyTorchRegressor(lr, gamma, epochs, self.model, criterion, batch_size)
        regr.fit(feature, target)
        self.model = regr.model

    def _choose_loss(self):
        name = self.config.train.loss
        error_parameter = self.config.train.error_parameter[0]

        if name.lower() in ["huber_loss", "huberloss"]:
            model = torch.nn.HuberLoss(delta=error_parameter.get("delta", 5e-2))
        elif name.lower() in ["relhuber", "rel_huber", "relhuber_loss", "relhuberloss"]:
            model = (
                RelHuberLoss(
                    delta=error_parameter.get("delta", 5e-3),
                    relative_delta=error_parameter.get("rel_delta", 1e-4),
                ),
            )
        return model

    def _get_nn_params(self):
        params = self.config.train.parameter[0]
        lr = params.get("learning_rate", 0.001)
        gamma = params.get("gamma", 0.99)
        epochs = params.get("epochs", 90)
        batch_size = params.get("batch_size", 1024)
        return lr, gamma, epochs, batch_size

    def score(self, feature, target):
        y_pred = self.predict(feature)
        return r2_score(y_true=target, y_pred=y_pred)
