from __future__ import annotations

import time

import numpy as np
from sklearn.model_selection import train_test_split

from hess_ml.src.data_generation import DataGeneration
from hess_ml.src.decorator.decorator import initProcess
from hess_ml.src.io import Input
from hess_ml.src.model_template import Training
from hess_ml.src.parser import Parser
from hess_ml.src.predicting import Predicting
from hess_ml.src.config import Configurations

class Environment(DataGeneration, Parser, Predicting, Input):
    def __init__(self):
        Predicting.__init__(self)
        DataGeneration.__init__(self)
        Parser.__init__(self)
        Input.__init__(self)

    def set_config(self):
        """
        Set config from the input toml and fills up the missing information with the default configs
        """
        self.config = Configurations(self.parsed_config)

    def print_config(self):
        print("Used config is:")
        print("")
        config_classe = ["general","molecule","train","predict"]
        for config_class in config_classe:
            print(f"{config_class.capitalize()} configurations:")
            class_vars:dict = vars(getattr(self.config,config_class))

            for name in class_vars.keys():
                
                if isinstance(class_vars[name], dict):
                    print("")
                    print(name)
                    for key in class_vars[name].keys():
                        print(key, ":",  class_vars[name][key])
                else:
                    print(name, ":", class_vars[name])

            print("")

    def import_data(self):
            self.gen_features()

    @initProcess
    def train_procedure(self):
        test_size_threshold = 0.0

        train_size = self.config.train.train_size

        self.rnd_states = self.config.general.random_state

        self.model_name = self.config.train.model_name

        self.runtype = self.config.general.runtype

        i = None 

        if isinstance(train_size, list):

            train_size = sorted(train_size)[::]

            for i in range(len(train_size)):

                rnd_seed = self.rnd_states

                self.shuffle_idx = np.arange(len(self.Features))

                print(
                    f"Percentage of data set used for training: {train_size[i]*100} %",
                )

                temp_time_old = time.time()

                train_size_temp = train_size[i] / train_size[-1]

                if not(train_size_temp > 1.0 - 1e-8):
                    self.shuffle_idx, temp = train_test_split(
                        self.shuffle_idx,
                        train_size=train_size_temp,
                        random_state=self.config.general.random_state,
                    )

                    del temp

                print(f"Total training points:{len(self.shuffle_idx)}")

                model_trainer = Training(self.config.train,self.config.general.threads)
                model_trainer.set_rnd_seed(rnd_seed)
                model_trainer.set_split(i)
                model_trainer.build_model()
                model_trainer.training(features=self.Features,
                                            targets=self.Targets,
                                            shuffle_idx=self.shuffle_idx,
                                            )
                del model_trainer

                if self.config.train.test_size > test_size_threshold:

                    self.predict(self.test_geo, folder=f"Model{i}_{rnd_seed}/")

                    self.error_estimation(
                        self.test_geo,
                        self.config.general.random_state,
                        train_size[i],
                    )

        else:
            print(
                f"Percentage of data set used for training: {train_size*100} %",
            )

            self.shuffle_idx = np.arange(len(self.Features))

            print(len(self.shuffle_idx))

            temp_time_old = time.time()

            model_trainer = Training(self.config.train,self.config.general.threads)
            model_trainer.set_rnd_seed(self.rnd_states)
            model_trainer.set_split(i)
            model_trainer.build_model()
            model_trainer.training(features=self.Features,
                                        targets=self.Targets,
                                        shuffle_idx=self.shuffle_idx)
            temp_time_new = time.time()

            print(f"Training was done in {round(temp_time_new - temp_time_old)} s")

            if self.config.train.test_size > test_size_threshold:
                temp_time_old = time.time()

                self.predict(self.test_geo)

                self.error_estimation(
                    self.test_geo,
                    self.config.general.random_state,
                    train_size,
                )

                temp_time_new = time.time()

                print(
                    f"Testing was done in {temp_time_new - temp_time_old: 0.2f} s",
                )

    @initProcess
    def prediction_procedure(self):
        self.folders = []

        if self.config.predict.folder is not None:
            self.parse_data_set(self.config.predict.folder)

        if self.config.predict.folder_list is not None:
            files = self.rd_txt_file(self.config.predict.folder_list)

            self.folders.extend(files)

        self.model_name = self.config.predict.model_name

        print(f"Starting prediction of {len(self.folders)} files")

        temp_time_old = time.time()

        self.predict(self.folders)

        temp_time_new = time.time()

        print(f"Prediction was done in {round(temp_time_new - temp_time_old)} s")


    def gen_features(self):
        if self.config.molecule.feature.lower() == "tblite":

            self.parse_data_set(self.config.molecule.folder)

            self.do_preparation_split(
                self.folders,
                len(self.folders),
                train_size=self.config.train.train_size,
                test_size=self.config.train.test_size,
                rnd_seed=self.config.general.random_state,
            )

            self.Targets = []
            self.Features = []

            self.generate_data(self.train_idx)

            self.Targets = np.array(self.Targets)
            self.Features = np.array(self.Features).astype(np.float32)

            np.savetxt(f"Features{self.n_split}.txt", self.Features)
            np.savetxt(f"Targets{self.n_split}.txt", self.Targets)

        elif self.config.molecule.feature.lower() == "numpy":
            print("Features and Targets are import from txt files.")

            self.Features = np.loadtxt("Features.txt",dtype=np.float32)
            self.Targets = np.loadtxt("Targets.txt")
            self.test_geo =  Input().rd_txt_file("test_files.txt")


        else:
            print("Feature generation must be specified.")
        # one could add different features that will be imported
