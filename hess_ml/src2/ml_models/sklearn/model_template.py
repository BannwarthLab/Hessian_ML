from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import numpy as np
import torch as torch 
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

from copy import deepcopy

from hess_ml.src.config import TrainConfig
from hess_ml.src.decorator.decorator import checkTiming, initProcess
from hess_ml.src.io import Input, Output
from hess_ml.src2.ml_models.sklearn.torch_models import PyTorchRegressor
import hess_ml.src2.ml_models.sklearn.torch_models as nn_hessian 
from hess_ml.src2.ml_models.sklearn.custom_loss_functions import CustomHuberLoss,CustomRelativeError
from hess_ml.src2.governance.globals import NUM_THREADS

from sklearn.decomposition import PCA


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

    def __init__(self,train_config:TrainConfig)-> None:
        self.train_config = train_config
        self.model_info = []

    def build_model(self):

        self.set_scaler(self.train_config.scale)
        self.set_selection(self.train_config.select)
        self.set_model(self.train_config.method)
        self.set_pipeline()
        self.set_target_transform(self.train_config.transform)

    @checkTiming(enabled=True)
    def training(self,features,targets,shuffle_idx):

        if self.method.lower() not in  ["hgbr",
                                        "mlpr",
                                        "hessiann",
                                        "red_hessiann",
                                        "custom_loss_hessiann",
                                        "custom_loss_hessiann2",
                                        "extfeat_hessiann",
                                        "custom_hessiann",
                                        "deep_hessiann"]:

            self.complete_model.set_params(regressor__n_jobs=NUM_THREADS)
            self.print_params()

            self.complete_model.fit(
                features[shuffle_idx],
                targets[shuffle_idx],
            )

        elif "nn" in self.method.lower():
            self.print_params()
            self.complete_model.fit(
                features[shuffle_idx],
                targets[shuffle_idx],
            )

        else:
            with parallel_backend("multiprocessing", n_jobs=NUM_THREADS):
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
        
        print("Setting selection")

        if selection.lower() in ["svd","truncatedsvd"]:
            feature_selector =  TruncatedSVD(n_components=300)
            self.model_info.append(("feature_selector", feature_selector))

        elif selection.lower() in ["variancethreshold"]:
            feature_selector = VarianceThreshold()
            self.model_info.append(("feature_selector", feature_selector))
        
        elif selection.lower() in ["pca"]:
            feature_selector = PCA(n_components=0.999,svd_solver='auto')
            self.model_info.append(("feature_selector", feature_selector))

        else:

            print("Selection Method was not recognized. Choose a different method.")

        return

    def set_scaler(self,scaling:str | None=None)-> list:
        """
        Define the standard scaler for the ML model.
        """

        if scaling is None:
            return
        
        print("Setting scaler")

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

        elif method == "hessiann":
                lr,gamma,epochs,batch_size = self.get_nn_params(params)
                model,criterion = self._hessiann_models(params,self.train_config.error_parameter[0])
                self.train_config
                regr_model = PyTorchRegressor(lr,gamma,epochs,model,criterion,batch_size)

        else:
            print("Model does not match to the existing models.")
            sys.exit()

        self.model_info.append(("regressor",regr_model))

        return self.model_info
    
    def _hessiann_models(self,params:dict,error_parameter:dict={}):

        model_key:str = params.get("model","CustomNN")
        model_key = model_key[:-2].lower().capitalize() + model_key[-2:].upper()

        criterion_key:str = params.get("criterion","huberloss")

        model = getattr(nn_hessian,model_key)

        criterions = {
            "relative_error" : CustomRelativeError(delta=error_parameter.get("delta",5e-3),
                                                   relative_delta=error_parameter.get("rel_delta",1e-4)),
            "custom_huber_loss" : CustomHuberLoss(),
            "huberloss" : torch.nn.HuberLoss(delta=error_parameter.get("delta",5e-2))
        }

        print(criterions[criterion_key.lower()])
        return model(), criterions[criterion_key.lower()]

    def get_nn_params(self,params:dict):
        lr = params.get("learning_rate",0.001)
        gamma = params.get("gamma",0.99)
        epochs = params.get("epochs",90)
        batch_size =  params.get("batch_size",1024)
        return lr,gamma,epochs,batch_size
    
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
        print(param_temp)
        for param in param_temp:
            print(f"{param}: {param_temp[param]}")


class ActiveTraining(Training):
    """A class to do active learning."""

    @checkTiming(enabled=True)
    def training(self,features,targets,shuffle_idx):

        train_shuffle_idx,test_shuffle_idx = train_test_split(shuffle_idx,train_size=0.2,random_state=self.rnd_seed)

        if self.method.lower() not in  ["hgbr","mlpr","hessiann","red_hessiann","custom_loss_hessiann","custom_loss_hessiann2","extfeat_hessiann","custom_hessiann","deep_hessiann"]:
            self.complete_model.set_params(regressor__n_jobs=NUM_THREADS)
            self.print_params()

            print("Starting first fit procedure.")
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
            print("Starting second fit procedure.")
            
            self.complete_model.fit(
                features[train_shuffle_idx],
                targets[train_shuffle_idx],
            )

        else:
            self.print_params()

            with parallel_backend("multiprocessing", n_jobs=NUM_THREADS):
                print("Starting first fit procedure.")
                self.complete_model.fit(
                    features[shuffle_idx],
                    targets[shuffle_idx],
                )

                pred_vals = self.complete_model.predict(features[test_shuffle_idx])
                err = pred_vals-targets[test_shuffle_idx]
                err_MAD = np.mean(np.abs(err))

                add_idx = test_shuffle_idx[np.sum(np.abs(err)/9,axis=1)>err_MAD*0.5]

                train_shuffle_idx = list(train_shuffle_idx) + list(add_idx)
                print("Starting second fit procedure.")
                self.complete_model.fit(
                    features[train_shuffle_idx],
                    targets[train_shuffle_idx],
                )
        print(
            f"Score on training data: {self.complete_model.score(features[shuffle_idx],targets[shuffle_idx])}",
        )

        self.dump_model()