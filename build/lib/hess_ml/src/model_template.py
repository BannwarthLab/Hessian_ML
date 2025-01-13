from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING, Optional

import numpy as np
from joblib import dump, parallel_backend
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVR

from hess_ml.src.config import TrainConfig
from hess_ml.src.decorator.decorator import checkTiming, initProcess
from hess_ml.src.io import Input, Output

if TYPE_CHECKING:
    from hess_ml.src.config import TrainConfig


class MyTransform(BaseEstimator,TransformerMixin):
    def fit(self, *_, **__):
        return self

    def transform(self, X):
        return (X+2)/3#np.sign(X)*np.log10(np.abs(X))-np.sign(X)#

    def inverse_transform(self, X):
        return X*3-2#-np.sign(X)*10**(-np.sign(X)*X+1)

class Training(Input,Output):

    def __init__(self,train_config:TrainConfig,threads=1)-> None:
        self.train_config = train_config
        self.model_info = []
        self.threads = threads

    def build_model(self):
        self.set_selection(self.train_config.select)
        self.set_scaler(self.train_config.scale)
        self.set_model(self.train_config.method)
        self.set_pipeline()
        self.set_target_transform(self.train_config.transform)

    @checkTiming(enabled=True)
    def training(self,features,targets,shuffle_idx):

        if self.method.lower() not in  ["hgbr","mlpr"]:
            self.complete_model.set_params(regressor__n_jobs=self.threads)
            self.print_params()

            self.complete_model.fit(
                features[shuffle_idx],
                targets[shuffle_idx],
            )

        else:
            self.print_params()

            with parallel_backend("multiprocessing", n_jobs=self.threads):
                self.complete_model.fit(
                    features[shuffle_idx],
                    targets[shuffle_idx],
                )

        print(
            f"Score on training data: {self.complete_model.score(features[shuffle_idx],targets[shuffle_idx])}",
        )

        self.dump_model()


    def set_target_transform(self,transformation:str | None=None):
        """
        Currently does not work. First test with this transformation were not promising.
        RMSD > 2 for 10 % train set in gdb7

        """

        if transformation is None:
            pass

        elif transformation:
            temp = TransformedTargetRegressor(regressor=self.complete_model,
                                                transformer=MyTransform(),
                                                check_inverse=False)

            self.complete_model = temp


    def set_selection(self,selection:str | None=None) -> list:
        """
        Define the feature selector for the ML model.
        param:
        selection:str: Defines the model needed
        processing_info:list: List of information to give to a pipiline afterwards
        """

        if selection is None:
            return

        if selection.lower() in ["svd","truncatedsvd"]:
            feature_selector =  TruncatedSVD(n_components=250)
            self.model_info.append(("feature_selector", feature_selector))

        if selection.lower() in ["variancethreshold"]:
            feature_selector = VarianceThreshold()
            self.model_info.append(("feature_selector", feature_selector))

        return

    def set_scaler(self,scaling:str | None=None)-> list:

        """
        Define the standard scaler for the ML model.
        """

        if scaling is None:
            return

        if scaling.lower() =="standardscaler":
            scaler = StandardScaler()
            self.model_info.append(("scaler",scaler))

        if scaling.lower() in ["normalizer"]:
            scaler = Normalizer()
            self.model_info.append(("scaler", scaler))
        return


    def set_model(self,method:str) -> list:
        """
        Define the ML model.
        """
        self.method = method

        params= self.train_config.parameter[0]

        method = method.lower()

        if method == "etr":
            regr_model = ExtraTreesRegressor(random_state=self.rnd_seed)

            regr_model.set_params(**params)

        elif method == "rfr":
            regr_model = RandomForestRegressor(random_state=self.rnd_seed)

            regr_model.set_params(**params)

        elif method == "svr":
            single_regr_model = SVR()

            single_regr_model.set_params(**params)

            regr_model = MultiOutputRegressor(single_regr_model)

        elif method == "mlpr":
            regr_model = MLPRegressor(hidden_layer_sizes=(500,500))

            regr_model.set_params(**params)

        elif method == "hgbr":

            single_regr_model = HistGradientBoostingRegressor(random_state=self.rnd_seed)
            single_regr_model.set_params(**params)

            regr_model = MultiOutputRegressor(single_regr_model)


        else:
            print("Model does not match to the existing models.")
            sys.exit()

        self.model_info.append(("regressor",regr_model))

        return self.model_info


    def set_pipeline(self):
        """
        Define the complete pipeline
        """

        self.complete_model = Pipeline(steps=self.model_info)

    def set_split(self,split=None):
        self.i_split = split

    def set_rnd_seed(self,seed):
        self.rnd_seed = seed

    def dump_model(self):

        model_name = self.train_config.model_name

        if self.i_split is not None:
            os.mkdir(f"Model{self.i_split}_{self.rnd_seed}")
            pathname = f"Model{self.i_split}_{self.rnd_seed}/{model_name}.joblib"
        else:
            pathname = f"{model_name}.joblib"

        dump(self.complete_model, pathname)

        print(f"Model is saved in {pathname}.\n")

    def print_params(self):

        print("Parameters for the Model:")
        param_temp = self.complete_model.get_params()
        for param in param_temp:
            print(f"{param}: {param_temp[param]}")


class ActiveTraining(Training):
    """A class to do active learning."""

    @checkTiming(enabled=True)
    def training(self,features,targets,shuffle_idx):

        train_shuffle_idx,test_shuffle_idx = train_test_split(shuffle_idx,train_size=0.2,random_state=self.rnd_seed)

        if self.method.lower() not in  ["hgbr","mlpr"]:
            self.complete_model.set_params(regressor__n_jobs=self.threads)
            self.print_params()

            self.complete_model.fit(
                features[train_shuffle_idx],
                targets[train_shuffle_idx],
            )

            pred_vals = self.complete_model.predict(features[test_shuffle_idx])

            deviations = np.sum(np.abs(pred_vals-targets[test_shuffle_idx]),axis=1)

            n = int(len(deviations)*0.6)

            add_idx = np.array(test_shuffle_idx)[np.argsort(np.abs(deviations))[-n:]]

            train_shuffle_idx = list(train_shuffle_idx) + list(add_idx)

            print(len(train_shuffle_idx))

            self.set_pipeline()

            self.complete_model.fit(
                features[train_shuffle_idx],
                targets[train_shuffle_idx],
            )

        else:
            self.print_params()

            with parallel_backend("multiprocessing", n_jobs=self.threads):
                self.complete_model.fit(
                    features[shuffle_idx],
                    targets[shuffle_idx],
                )

                pred_vals = self.complete_model.predict(features[test_shuffle_idx])
                err = pred_vals-targets[test_shuffle_idx]
                err_MAD = np.mean(np.abs(err))

                add_idx = test_shuffle_idx[np.sum(np.abs(err)/9,axis=1)>err_MAD*0.5]

                train_shuffle_idx = list(train_shuffle_idx) + list(add_idx)

                self.complete_model.fit(
                    features[train_shuffle_idx],
                    targets[train_shuffle_idx],
                )
        print(
            f"Score on training data: {self.complete_model.score(features[shuffle_idx],targets[shuffle_idx])}",
        )

        self.dump_model()
