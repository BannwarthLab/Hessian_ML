import pytest

import os
 
from pathlib import Path
import numpy as np
from mlhess.scripts.data_handling import FittingDataHandler
from mlhess.management.config import Configurator
from mlhess.management.base_settings import PROCESSED_DATA_FOLDER

from mlhess.machinelearning.target.hessian import NuclearHessian, NuclearHessianPM
from mlhess.machinelearning.feature.base_class import Feature


os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libgomp.so.1"

files = ["test_files.txt", "train_files.txt", "not_considered"]
folders = [PROCESSED_DATA_FOLDER]


def test_run_protocol(cleanup_file, cleanup_folder):
    inp_fname = "collector_data/input.toml"

    for file in files:
        cleanup_file(file)
    for folder in folders:
        cleanup_folder(folder)

    path = os.path.join(Path(__file__).parent, inp_fname)
    cwd = os.path.abspath('./')
    os.chdir(os.path.join(Path(__file__).parent, "collector_data"))
    config = Configurator(path)
    fdh = FittingDataHandler(config)
    fdh.run_protocol()

    assert os.path.isdir(PROCESSED_DATA_FOLDER)
    assert len(fdh.Features) == 44
    np.testing.assert_array_equal(fdh.train_idx, np.array([0, 1, 3]))

    os.chdir(cwd)
    
def test_collect_folder():
    inp_fname = "collector_data/input.toml"
    path = os.path.join(Path(__file__).parent, inp_fname)
    config = Configurator(path)
    fdh = FittingDataHandler(config)
    fdh.collect_folders()

    folder_names = []
    for i in range(1, 5):
        folder_names.append(f"collector_data/mol_structures/{i:04d}")

    for true_folder, test_folder in zip(folder_names, fdh.folder_names):
        assert true_folder == test_folder


def test_do_preparation_split(cleanup_file):
    inp_fname = "collector_data/input.toml"
    for file in files:
        cleanup_file(file)

    path = os.path.join(Path(__file__).parent, inp_fname)
    config = Configurator(path)
    fdh = FittingDataHandler(config)

    fdh.folder_names = ["0", "1", "2", "3"]

    fdh.do_preparation_split()

    for geo_true, geo_test in zip(["0", "1", "3"], fdh.train_geo):
        assert geo_true == geo_test


@pytest.mark.parametrize(
    "target_model, expected", [("mlh", NuclearHessian), ("mlh_pm", NuclearHessianPM)]
)
def test_pick_target_type(target_model, expected):
    config = Configurator(None)
    fdh = FittingDataHandler(config)
    fdh.config.train.target_model = target_model

    fdh.folder_names = ["0", "1", "2", "3"]
    target_type = fdh._pick_target_class()
    assert type(target_type) is type(expected)


@pytest.mark.parametrize(
    "feature_model, expected", [("train", Feature), ("predict", Feature)]
)
def test_pick_feature_class(feature_model, expected):
    config = Configurator(None)
    fdh = FittingDataHandler(config)
    fdh.config.collector.feature_type = feature_model
    fdh.folder_names = ["0", "1", "2", "3"]

    target_type = fdh._pick_feature_class()

    assert type(target_type) is type(expected)


@pytest.mark.parametrize("folder_name, nat", [("single_mol_data/0001", 5)])
def test_collect_feature_target_from_mol(folder_name, nat):
    path = os.path.join(Path(__file__).parent, folder_name)
    cwd = os.path.abspath("./")
    os.chdir(os.path.join(Path(__file__).parent, folder_name))

    config = Configurator(None)
    config.molecule.xyz_file = "xtbopt.xyz"
    fdh = FittingDataHandler(config)

    fdh.n_data = 0
    fdh.Targets = []
    fdh.Features = []

    mol = fdh.collect_feature_target_from_mol(path)
    assert mol.nat == nat
    assert mol.hessian.hessian.shape[0] == nat * 3
    assert fdh.n_data == nat * (nat - 1) / 2
    assert mol.calc_succeeded is True
    os.chdir(cwd)

def test_dump_features_and_targets(cleanup_folder):
    cleanup_folder(PROCESSED_DATA_FOLDER)
    config = Configurator("default")
    fdh = FittingDataHandler(config)
    os.mkdir(PROCESSED_DATA_FOLDER)

    fdh.n_split = 10
    fdh.Features = np.random.random([10, 4])
    fdh.Targets = np.random.random([10, 1])

    fdh._dump_features_and_targets(del_ft=True)

    assert fdh.split is True
    assert fdh.Features == []


def test_collect_processed_feature_targets(cleanup_folder):
    cleanup_folder(PROCESSED_DATA_FOLDER)
    config = Configurator("default")
    fdh = FittingDataHandler(config)
    os.mkdir(PROCESSED_DATA_FOLDER)

    fdh.n_split = 10
    temp_feat = np.float32(np.random.random([10, 4]))
    temp_target = np.float32(np.random.random([10, 1]))
    fdh.Features = temp_feat
    fdh.Targets = temp_target

    fdh._dump_features_and_targets(del_ft=True)
    fdh._collect_processed_feature_targets()

    np.testing.assert_array_equal(fdh.Features, temp_feat)
    np.testing.assert_array_equal(fdh.Targets, temp_target)
