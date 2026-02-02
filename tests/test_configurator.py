import pytest
from mlhess.management.config import Configurator


@pytest.mark.parametrize(
    "key, expected",
    [
        ("feature_class", "xtbml"),
        ("target_class", "mlh"),
        ("xyz_file", "xtbopt.xyz"),
        ("target_file", "hessian"),
        ("solvent", None),
        ("alphas", [1.0]),
    ],
)
def test_molecule_configurator(key, expected):
    config = Configurator(None)
    sub_config = config.molecule
    actual = getattr(sub_config, key)
    assert actual == expected


@pytest.mark.parametrize(
    "key, expected",
    [
        ("folder", None),
        ("file_list", None),
        ("gather", "molecules"),
    ],
)
def test_collector_configurator(key, expected):
    config = Configurator(None)
    sub_config = config.collector
    actual = getattr(sub_config, key)
    assert actual == expected


@pytest.mark.parametrize(
    "key, expected",
    [
        ("threads", 4),
        ("gpu", True),
        ("random_state", 79),
        ("runtype", "mlhess"),
        ("collector", True),
        ("trainer", True),
        ("predictor", False),
    ],
)
def test_general_configurator(key, expected):
    config = Configurator(None)
    sub_config = config.general
    actual = getattr(sub_config, key)
    assert actual == expected


@pytest.mark.parametrize(
    "key, expected",
    [
        ("target_model", "mlh_l"),
        ("loss", "relHub"),
        ("train_size", 0.75),
        ("validation_size", 0.1),
        ("test_size", 0.25),
        ("method", "ETR"),
        ("model_name", "MLH"),
        ("parameter", [{}]),
        ("error_parameter", [{}]),
        ("active", False),
        ("select", None),
        ("scale", None),
        ("transform", None),
    ],
)
def test_train_configurator(key, expected):
    config = Configurator(None)
    sub_config = config.train
    actual = getattr(sub_config, key)
    assert actual == expected


@pytest.mark.parametrize(
    "key, expected",
    [
        ("train", None),
    ],
)
def test_internal_configurator(key, expected):
    config = Configurator(None)
    sub_config = config.internal
    actual = getattr(sub_config, key)
    assert actual == expected
