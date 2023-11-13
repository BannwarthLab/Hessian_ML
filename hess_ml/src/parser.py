import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import tomli


class Parser:
    def __init__(self) -> None:
        self.cwd = os.getcwd()
        self.LoadDefaultConfig()

    def parse(self) -> None:
        self.arg_parser = ArgumentParser()

        self.arg_parser.add_argument(
            "-i",
            "--input",
            type=str,
            default="default",
            help="INPUT must be a .toml file",
        )

        self.arg_parser.add_argument(
            "-to",
            "--toml",
            action="store_true",
            help="generates a basic example .toml file",
        )

    def LoadDefaultConfig(self) -> None:
        abs_path = Path(__file__).parent / "default_input/input.toml"

        with open(abs_path, mode="rb") as fp:
            self.default_config = tomli.load(fp)

    def parse_toml(self) -> None:
        """

        Parser for the .toml to generate a dictionoary for the input information.

        """

        inp_file = self.arg_parser.parse_args().input

        if os.path.isfile(inp_file):
            with open(self.arg_parser.parse_args().input, mode="rb") as fp:
                self.config = tomli.load(fp)
                print(f"Input file: {inp_file} is used.")

        elif inp_file == "default":
            if os.path.isfile("input.toml"):
                with open("input.toml", mode="rb") as fp:
                    self.config = tomli.load(fp)

                    print("Input file: input.toml is used.")
            else:
                print("Default setup is used.")

                abs_path = Path(__file__).parent / "default_input/input.toml"

                with open(abs_path, mode="rb") as fp:
                    self.config = tomli.load(fp)

        else:
            print(f"Input file: {inp_file} not found.")
            print("")

            sys.exit()

    def get_config(self) -> dict:
        return self.config

    def parse_data_set(self, main_folder) -> None:
        print(f"Parsing data set in {main_folder}...", end="")
        xyz_file = self.config.get("molecule", {"xyz_file": "xtbopt.xyz"}).get(
            "xyz_file",
        )

        target_file = self.config.get("molecule", {"target_file": "hessian"}).get(
            "target_file",
        )
        walker = os.walk(main_folder)

        self.folders = []

        for folder in walker:
            if xyz_file in folder[-1] and target_file in folder[-1]:
                self.folders.append(folder[0])

        self.folders = sorted(self.folders)

        print("done")

        print(f"Total of {len(self.folders)} folders found.")
        print("")

    # def parse_general(self):
    #     """
    #     Sets the general parameter needed for:
    #         - the generation of features
    #         - the training
    #         - the prediction
    #     """

    #     self.feature_file = self.config["general"].get("feature", "ml_feature.csv")

    #     self.target_file = self.config["general"].get(
    #         "target_file", self.runtype_target[self.config["runtype"]]
    #     )

    #     self.xyz_file = self.config["general"].get("xyz_file", "xtbopt.xyz")

    #     self.gradient_file = self.config["general"].get("gradient_file", "gradient")

    #     self.folder = self.config["general"].get("folder", None)

    #     self.rnd_seed = self.config["general"].get(
    #         "random_seed", np.random.randint(0, 1000)
    #     )

    #     return

    # def parse_feature(self):
    #     """
    #     Sets the parameter needed for the generation of features
    #     """

    #     self.feature_gen = self.config["feature"].get("generate", True)

    #     if not (self.feature_gen):
    #         self.feature_import = self.config["feature"].get("import", "numpy")

    #     # Implement tblite api for generation of the basic features
    #     # self.tblite = self.config['feature'].get('tblite',False)
    #     # If true they are generated with tbltie in advance

    #     return

    # def parse_training(self):
    #     """
    #     Sets the parameter needed for the training of the ML Model
    #     """

    #     self.train_size = self.config["training"].get("train_size", 0.75)

    #     if type(self.train_size) == list:
    #         train_max = max(self.train_size)

    #     else:
    #         train_max = self.train_size

    #     self.test_size = self.config["training"].get("test_size", 1 - train_max)

    #     self.method = self.config["training"].get("method", "ETR")

    #     self.SearchCV = self.config["training"].get("SearchCV", "None")

    #     self.testing = self.config["training"].get("test", False)

    #     self.selection = self.config["training"].get("selection", False)

    #     if self.SearchCV.lower() == "random":
    #         self.n_iter_search = self.config["training"].get("n_iter", 25)

    #     self.model_name = self.config["training"].get(
    #         "model_name", f"{self.runtype_target[self.config['runtype']]}_model"
    #     )

    #     try:
    #         if "hidden_layer_sizes" in self.config["training"]["parameter"].keys():
    #             self.config["training"]["parameter"]["hidden_layer_sizes"] = tuple(
    #                 self.config["training"]["parameter"]["hidden_layer_sizes"]
    #             )

    #     except:
    #         print(
    #             "No parameters for the model are specified. Default parameters are set"
    #         )

    #         self.config["training"]["parameter"] = {}

    #     return

    # def parse_predict(self):  # Set parameters for prediction
    #     """
    #     Sets the parameter needed for the prediction of a set of systems
    #     """

    #     self.predict_folder = self.config["predict"].get("folder", False)

    #     if self.config.get("training", False):
    #         self.model_name = self.config["predict"].get(
    #             "model",
    #             self.config["training"].get(
    #                 "model_name", f"{self.runtype_target[self.config['runtype']]}_model"
    #             ),
    #         )
    #     else:
    #         self.model_name = self.config["predict"].get(
    #             "model", f"{self.runtype_target[self.config['runtype']]}_model"
    #         )

    #     self.predict_model_folder = self.config["predict"].get("model_folder", False)

    #     self.predict_subfolder = self.config["general"].get("subfolder", False)

    #     self.predict_files = self.config["predict"].get("files", None)

    #     self.predict_data_gen = self.config["predict"].get("generate", True)

    #     self.selection = self.config["predict"].get("selection", False)

    #     self.normalization = self.config["predict"].get("normalization", False)

    #     if self.predict_folder == self.folder:
    #         print(
    #             "You chose the same folder for prediction as you chose for training and testing. This is not recommended."
    #         )

    #     return
