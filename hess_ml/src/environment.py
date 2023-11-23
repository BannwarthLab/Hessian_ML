import time

import numpy as np
from sklearn.model_selection import train_test_split

from hess_ml.src.data_generation import DataGeneration
from hess_ml.src.decorator.decorator import initProcess
from hess_ml.src.io import Input
from hess_ml.src.model_template import Training
from hess_ml.src.parser import Parser
from hess_ml.src.predicting import Predicting

# 0.09769007921573508
# 0.05527


class Environment(DataGeneration, Parser, Predicting, Input):
    def __init__(self):
        Predicting.__init__(self)
        DataGeneration.__init__(self)
        Parser.__init__(self)
        Input.__init__(self)

    def set_general_config(self):
        """
        Set config from the input toml and fills up the missing information with the default configs
        """

        for MainKey in ["general", "molecule", "train", "predict"]:
            if MainKey in self.config:
                for key in self.default_config[MainKey]:
                    if key not in self.config[MainKey]:
                        self.config[MainKey][key] = self.default_config[MainKey][key]

            elif MainKey == "general":
                self.config[MainKey] = self.default_config[MainKey]

        self.rnd_seed = self.config["general"]["random_state"]
        self.threads = self.config["general"]["threads"]

    def print_config(self):
        print("Used config is:")
        print("")
        for dicts in self.config:
            if isinstance(self.config[dicts], dict):
                print("")
                print(dicts)
                for dict_ in self.config[dicts]:
                    print(dict_, ":", self.config[dicts][dict_])

            else:
                print(dicts, ":", self.config[dicts])

        print("")

    def import_data(self):
            self.gen_features()



    @initProcess
    def train_procedure(self):
        test_size_threshold = 0.0

        self.train_size = self.config["train"]["train_size"]

        self.model_name = self.config["train"]["model_name"]

        self.runtype = self.config["general"]["runtype"]

        if isinstance(self.train_size, list):
            self.train_size = sorted(self.train_size)[::]

            for i in range(len(self.train_size)):
                self.shuffle_idx = np.arange(len(self.Features))

                print(
                    f"Percentage of data set used for training: {self.train_size[i]*100} %",
                )

                temp_time_old = time.time()

                train_size = self.train_size[i] / self.train_size[-1]

                if not(train_size > 1.0 - 1e-8):
                    self.shuffle_idx, temp = train_test_split(
                        self.shuffle_idx,
                        train_size=train_size,
                        random_state=self.config["general"]["random_state"],
                    )

                    del temp

                print(f"Total training points:{len(self.shuffle_idx)}")

                model_trainer = Training(self.config["train"],self.rnd_seed,self.threads)
                model_trainer.build_model()
                model_trainer.training(features=self.Features,
                                            targets=self.Targets,
                                            shuffle_idx=self.shuffle_idx,
                                            i_split=i,
                                            )
                del model_trainer

                if self.config["train"]["test_size"] > test_size_threshold:

                    self.predict(self.test_geo, folder=f"Model{i}/")

                    self.error_estimation(
                        self.test_geo,
                        self.config["general"]["random_state"],
                        self.train_size[i],
                    )

        else:
            print(
                f"Percentage of data set used for training: {self.train_size*100} %",
            )

            self.shuffle_idx = np.arange(len(self.Features))

            print(len(self.shuffle_idx))

            temp_time_old = time.time()

            model_trainer = Training(self.config["train"],self.rnd_seed,self.threads)
            model_trainer.build_model()
            model_trainer.training(features=self.Features,
                                        targets=self.Targets,
                                        shuffle_idx=self.shuffle_idx)
            temp_time_new = time.time()

            print(f"Training was done in {round(temp_time_new - temp_time_old)} s")

            if self.config["train"]["test_size"] > test_size_threshold:
                temp_time_old = time.time()

                self.predict(self.test_geo)

                self.error_estimation(
                    self.test_geo,
                    self.config["general"]["random_state"],
                    self.train_size,
                )

                temp_time_new = time.time()

                print(
                    f"Testing was done in {temp_time_new - temp_time_old: 0.2f} s",
                )

    @initProcess
    def prediction_procedure(self):

        if self.config["predict"].get("folder", False):
            self.parse_data_set(self.config["predict"].get("folder"))

        if self.config["predict"].get("predict_list", False):
            files = self.rd_txt_file(self.config["predict"].get("predict_list"))

            self.folders.extend(files)

        self.model_name = self.config["predict"]["model_name"]

        print(f"Starting prediction of {len(self.folders)} files")

        temp_time_old = time.time()

        self.predict(self.folders)

        temp_time_new = time.time()

        print(f"Prediction was done in {round(temp_time_new - temp_time_old)} s")


    def gen_features(self):
        if self.config["molecule"]["feature"].lower() == "tblite":
            # Maybe adapt these infos in a set_config function
            self.parse_data_set(self.config["molecule"].get("folder"))

            self.do_preparation_split(
                self.folders,
                len(self.folders),
                train_size=self.config["train"]["train_size"],
                test_size=self.config["train"]["test_size"],
                rnd_seed=self.config["general"]["random_state"],
            )

            self.Targets = []
            self.Features = []

            self.generate_data(self.train_idx)

            self.Targets = np.array(self.Targets)#.astype(np.float32)
            self.Features = np.array(self.Features).astype(np.float32)

            np.savetxt("Features.txt", self.Features)
            np.savetxt("Targets.txt", self.Targets)

        elif self.config["molecule"]["feature"].lower() == "numpy":
            print("Features and Targets are import from txt files.")

            self.Features = np.loadtxt("Features.txt",dtype=np.float32)
            self.Targets = np.loadtxt("Targets.txt")
            self.test_geo =  Input().rd_txt_file("test_files.txt")


        else:
            print("Feature generation must be specified.")
        # one could add different features that will be imported
