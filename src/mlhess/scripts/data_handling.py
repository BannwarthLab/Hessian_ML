"""potentially rename to FittingDataHandler or similarly as Collector does not mean it is also for handling data."""

import os
import glob
import time
import sys
import numpy as np

from sklearn.model_selection import train_test_split
import mlhess.management.base_settings as base_settings
from mlhess.management.config import Configurator
from mlhess.machinelearning.feature.base_class import Feature
from mlhess.machinelearning.target.hessian import NuclearHessian, NuclearHessianPM

from mlhess.utils.io.reader import read_txt_file
from mlhess.utils.io.writer import list_to_txt
from mlhess.utils.io.parser import parse_data_set
from mlhess.utils.chemistry.molecule import Molecule
from mlhess.calculator.tblite_wrapper import Calculator
from mlhess.utils.patcher import patch_methods


class FittingDataHandler:
    def __init__(self, config: Configurator):
        self.config = config

    def collect_folders(self):
        """Parses data set folders with given information and adds a list of folders if given."""
        self.folder_names = parse_data_set(self.config)
        self.folder_names.extend(read_txt_file(self.config.collector.file_list))

    def do_preparation_split(self):
        """
        Does a split of the geometry file directories into train and test sets.
        Saves the information in txt files
        """

        train_size = self.config.train.train_size
        test_size = self.config.train.test_size
        rnd_seed = self.config.general.random_state

        max_train_size = 1.0
        total_structures = len(self.folder_names)
        geo_idx = np.arange(0, total_structures)

        train_size_temp = (
            max(train_size) if isinstance(train_size, list) else train_size
        )

        if train_size_temp == max_train_size:
            self.train_idx = geo_idx
            self.test_idx = []

        else:
            self.train_idx, self.test_idx = train_test_split(
                geo_idx,
                test_size=test_size,
                train_size=train_size_temp,
                random_state=rnd_seed,
            )

            self.comp_idx = np.concatenate((self.train_idx, self.test_idx), axis=None)

            geo_idx = geo_idx[self.comp_idx]

        self.test_geo = []

        for i in self.test_idx:
            self.test_geo.append(self.folder_names[i])

        self.train_geo = []

        for i in self.train_idx:
            self.train_geo.append(self.folder_names[i])

        list_to_txt(self.test_geo, os.path.join("", "test_files.txt"))

        list_to_txt(self.train_geo, os.path.join("", "train_files.txt"))

    def run_protocol(self) -> Configurator:
        """Collect data depending on given input

        :param config: Configurations
        :type config: Configurator
        :return: optionally adapted configurations
        :rtype: Configurator
        """

        if self.config.collector.gather.lower() in ["molecules"]:
            if not (os.path.isdir(base_settings.PROCESSED_DATA_FOLDER)):
                os.mkdir(base_settings.PROCESSED_DATA_FOLDER)

            self.collect_folders()

            self.do_preparation_split()

            self.generate_data(self.train_idx)

            if self.split:
                self._collect_processed_feature_targets(
                    tmp_folder=base_settings.PROCESSED_DATA_FOLDER
                )

        if self.config.collector.gather.lower() in [
            "npy",
            "numpy_split",
            "numpy",
            "processed",
        ]:
            self._collect_processed_feature_targets()

        return self.config

    def _collect_processed_feature_targets(self, tmp_folder=None):
        msg = "Features and Targets are import from .npy files."
        print(msg)
        if tmp_folder is None:
            tmp_folder = self.config.collector.folder

        if tmp_folder is None:
            tmp_folder = base_settings.PROCESSED_DATA_FOLDER

        if not (os.path.isdir(tmp_folder)):
            msg = f"Path {tmp_folder} not found. Exiting programme."
            print(msg)
            sys.exit()

        feature_files = glob.glob(os.path.join(tmp_folder, "Features**.npy"))
        target_files = glob.glob(os.path.join(tmp_folder, "Targets**.npy"))

        feature_files = sorted(feature_files)
        target_files = sorted(target_files)

        assert len(feature_files) == len(target_files), (
            "Mismatch of number of target and feature files."
        )

        for ffeature, ftarget in zip(feature_files, target_files):
            print(ffeature, ftarget)
            assert ffeature[-8:-4] == ftarget[-8:-4], (
                "Mismath of names of feature and target files."
            )

        with open(feature_files[0], "rb") as f:
            features = np.load(f)
            f.close()

        with open(target_files[0], "rb") as f:
            targets = np.load(f)
            f.close()

        for f_feature, f_target in zip(feature_files[1:], target_files[1:]):
            with open(f_feature, "rb") as f:
                features = np.append(features, np.load(f), axis=0)
                f.close()

            with open(f_target, "rb") as f:
                targets = np.append(targets, np.load(f), axis=0)
                f.close()

        self.Targets = targets
        self.Features = features

    def _pick_feature_class(self):
        """pick a class to describe features.

        :return: _description_
        :rtype: _type_
        """
        return Feature

    def _pick_target_class(self):
        """Pick the target type for further calculations.

        :return: target class
        :rtype: AbstractTarget
        """
        match self.config.molecule.target_class.lower():
            case "mlh":
                target_type = NuclearHessian
            case "mlh_pm":
                target_type = NuclearHessianPM

        return target_type

    def generate_data(self, idx=None, max_n_data=5e5):
        self.wall_time0 = time.time()

        print("Starting Data Generation for Features...")

        # ________Parallelized Feature Generation___________

        self.not_considered = []

        methods = {
            "feature_class": self._pick_feature_class(),
            "hess_class": self._pick_target_class(),
        }

        self.n_data = 0
        self.split = False
        self.n_split = 0

        self.Targets: list | np.ndarray = []
        self.Features: list | np.ndarray = []

        with patch_methods(Molecule, methods):
            for geo in idx:
                self.collect_feature_target_from_mol(dir=self.folder_names[geo])

                print(f"Number of DataPoints {self.n_data}")

                if self.n_data > max_n_data:
                    self._dump_features_and_targets(True)

            self._dump_features_and_targets(False)

            outputfile_name = "not_considered"
            with open(outputfile_name, "w") as outfile:
                outfile.write("\n".join(str(i) for i in self.not_considered))
            outfile.close()

            print("")
            print(
                f"Features and Targets of {len(idx) - len(self.not_considered)} structures "
                f"were generated in {round(time.time() - self.wall_time0)} s\n",
            )
            print(f"""Features and Targets of {len(self.not_considered)} structures were not considered. \n
                The location of these structures can be found in 'not_considered'.""")

    def _dump_features_and_targets(self, del_ft):
        self.Targets = np.array(self.Targets)
        self.Features = np.array(self.Features).astype(np.float32)

        print(f"Shape of features: {self.Features.shape}")

        with open(
            os.path.join(
                base_settings.PROCESSED_DATA_FOLDER, f"Features{self.n_split:04d}.npy"
            ),
            "wb",
        ) as f:
            np.save(f, self.Features)
            f.close()

        with open(
            os.path.join(
                base_settings.PROCESSED_DATA_FOLDER, f"Targets{self.n_split:04d}.npy"
            ),
            "wb",
        ) as f:
            np.save(f, self.Targets)
            f.close()

        self.n_split += 1

        if self.n_split > 1:
            self.split = True

        if del_ft:
            self.Features = []
            self.Targets = []
            self.n_data = 0

    def collect_feature_target_from_mol(self, dir: str):
        print(f"Path: {dir}")
        mol = Molecule(dir, self.config.molecule.xyz_file)

        print(f"Number of atoms: {mol.nat}")

        mol.read_hessian(self.config.molecule.target_file)
        mol.prepare_training(Calculator)

        self.n_data += mol.nat * (mol.nat - 1) / 2
        print(f"Feature shape: {mol.feature.processed.shape}")

        if mol.calc_succeeded:
            if np.isnan(np.sum(mol.processed_target)):
                print(mol.processed_target)
                print("Some feature is NaN.")
                sys.exit()

            self.Features.extend(mol.feature.processed)
            self.Targets.extend(mol.processed_target)

        else:
            self.not_considered.append(
                os.path.join(
                    self.config.collector.folder,
                    self.config.molecule.xyz_file,
                ),
            )

        return mol
