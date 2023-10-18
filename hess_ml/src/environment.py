import time

import numpy as np
from sklearn.model_selection import train_test_split

from hess_ml.src.data_generation import DataGeneration
from hess_ml.src.io import Input
from hess_ml.src.parser import Parser
from hess_ml.src.predicting import Predicting
from hess_ml.src.Training import Training

# 0.09769007921573508
# 0.05527


class Environment(DataGeneration, Parser, Training, Predicting, Input):
    def __init__(self):
        Training.__init__(self)  # initializes all parent classes
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
        np.random.default_rng(self.rnd_seed)

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
        if self.config["general"].get("feature"):
            self.gen_features()

    def do_train(self):
        test_size_threshold = 0.0

        if self.config["general"]["train"]:
            
            self.train_size = self.config["train"]["train_size"]

            self.model_name = self.config["train"]["model_name"]

            self.runtype = self.config["general"]["runtype"]

            if isinstance(self.train_size, list):
                self.train_size = sorted(self.train_size)[::]

                print(self.train_size)

                for i in range(len(self.train_size)):
                    self.shuffle_idx = np.arange(len(self.Features))

                    print(
                        f"Percentage of data set used for training: {self.train_size[i]*100} %",
                    )

                    temp_time_old = time.time()

                    # self.do_train_split(i)
                    train_size = self.train_size[i] / self.train_size[-1]
                    self.shuffle_idx, temp = train_test_split(
                        self.shuffle_idx,
                        train_size=np.clip(train_size, 0.0, 1.0 - 1e-8),
                        random_state=self.config["general"]["random_state"],
                    )

                    del temp

                    print(f"Total training strucutres:{len(self.shuffle_idx)}")

                    # self.import_FT()

                    self.TrainModel(i_split=i)

                    temp_time_new = time.time()

                    print(
                        f"Training was done in {round(temp_time_new - temp_time_old)} s",
                    )

                    if self.config["train"]["test_size"] > test_size_threshold:
                        temp_time_old = time.time()

                        self.predict(self.test_geo, folder=f"Model{i}/")

                        self.error_estimation(
                            self.test_geo,
                            self.config["general"]["random_state"],
                            self.train_size[i],
                        )

                        temp_time_new = time.time()

                        print(
                            f"Testing was done in {temp_time_new - temp_time_old: 0.2f} s",
                        )

            else:
                print(
                    f"Percentage of data set used for training: {self.train_size*100} %",
                )

                self.shuffle_idx = np.arange(len(self.Features))

                temp_time_old = time.time()

                self.TrainModel()

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

    def do_prediction(self):

        if self.config['general'].get("predict", False) and self.config.get("predict", False):
            
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

            self.Targets = np.array(self.Targets).astype(np.float32)
            self.Features = np.array(self.Features).astype(np.float32)

            np.savetxt("Features.txt", self.Features)
            np.savetxt("Targets.txt", self.Targets)

        elif self.config["molecule"]["feature"].lower() == "numpy":
            print("Features and Targets are import from txt files.")

            self.Features = np.loadtxt("Features.txt")
            self.Targets = np.loadtxt("Targets.txt")

        else:
            print("Feature generation must be specified.")
        # one could add different features that will be imported
