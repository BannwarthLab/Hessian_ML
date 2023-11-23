from __future__ import annotations

import os
import sys
import time
from typing import Optional

import numpy as np
from joblib import dump, parallel_backend
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVR

from hess_ml.src.decorator.decorator import checkTiming, initProcess
from hess_ml.src.io import Input, Output


class MyTransform(BaseEstimator,TransformerMixin):
    def fit(self, *_, **__):
        return self

    def transform(self, X):
        return (X+2)/3#np.sign(X)*np.log10(np.abs(X))-np.sign(X)#

    def inverse_transform(self, X):
        return X*3-2#-np.sign(X)*10**(-np.sign(X)*X+1)


class Training(Input,Output):

    def __init__(self,config:dict,seed=None,threads=1)-> None:
        self.config = config
        self.model_info = []
        self.rnd_seed = seed
        self.threads = threads


    def build_model(self):
        self.set_selection(self.config.get("select",None))
        self.set_scaler(self.config.get("scale",None))
        self.set_model(self.config.get("method","etr"))
        self.set_pipeline()
        self.set_target_transform(self.config.get("transform",None))


    @checkTiming(enabled=True)
    def training(self,features,targets,shuffle_idx,i_split=None):
        self.print_params()

        if self.method != "mlpr":
            self.complete_model.set_params(regressor__n_jobs=self.threads)
            self.complete_model.fit(
                features[shuffle_idx],
                targets[shuffle_idx],
            )

        else:

            with parallel_backend("multiprocessing", n_jobs=self.threads):
                self.complete_model.fit(
                    features[shuffle_idx],
                    targets[shuffle_idx],
                )

        print(
            f"Score on training data: {self.complete_model.score(features[shuffle_idx],targets[shuffle_idx])}",
        )

        self.dump_model(i_split=i_split)


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
            feature_selector =  TruncatedSVD(n_components=50, algorithm="arpack")
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
        params= self.config.get("parameter",[{}])[0]

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
            regr_model = MLPRegressor(hidden_layer_sizes=(465,465))

            regr_model.set_params(**params)

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



    def dump_model(self,i_split=None):

        model_name = self.config.get("model_name")

        if i_split is not None:
            os.mkdir(f"Model{i_split}")
            pathname = f"Model{i_split}/{model_name}.joblib"
        else:
            pathname = f"{model_name}.joblib"

        dump(self.complete_model, pathname)

        print(f"Model is saved in {pathname}.\n")



    def do_train_split(self, i):
        """
        Does a split of the geometry file directories into train and test sets.
        Saves the information in txt files
        """

        print(f"{self.train_size[i]*100} % of the set is used for training.")

        self.files, temp = train_test_split(
            self.train_geo,
            train_size=self.train_size[i],
            random_state=self.rnd_seed,
        )

        self.comp_idx = np.concatenate((self.train_idx, self.test_idx), axis=None)

        del temp

        mypath = f"Model{i}"

        if not os.path.isdir(mypath):
            os.makedirs(mypath)

        self.data_to_txt(self.train_geo, os.path.join(f"Model{i}/", "train_files.txt"))



    def print_params(self):

        print("Parameters for the Model:")
        param_temp = self.complete_model.get_params()
        for param in param_temp:
            print(f"{param}: {param_temp[param]}")

