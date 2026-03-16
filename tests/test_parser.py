import os
import pytest

from pathlib import Path
import mlhess.utils.io.parser as parser
from mlhess.management.config import Configurator


@pytest.mark.parametrize(
    "inp_fname, expected,exit",
    [("input.toml", dict, 0), ("wrong_name.toml", dict, 1), ("default", dict, 0)],
)
def test_parse_input_toml_file(inp_fname, expected, exit):
    if inp_fname != "default":
        full_path = os.path.join(Path(__file__).parent, "collector_data", inp_fname)
    else:
        full_path = "default"
    try:
        inputs = parser.parse_input_toml_file(full_path)
    except SystemExit as exc_info:
        err_code = exc_info.code
    else:
        err_code = 0

    assert err_code == exit
    if exit == 0:
        assert type(inputs) is expected


@pytest.mark.parametrize(
    "inp_fname, expected",
    [
        ("input.toml", 4),
        ("input_xyz_name_differs.toml", 0),
        ("input_hess_name_differs.toml", 0),
    ],
)
def test_parse_data_set(inp_fname, expected):
    full_path = os.path.join(Path(__file__).parent, "collector_data", inp_fname)
    config = Configurator(full_path)
    cwd = os.path.abspath("./")
    os.chdir(os.path.join(Path(__file__).parent,'collector_data'))
    folder_names = parser.parse_data_set(config)
    assert len(folder_names) == expected
    os.chdir(cwd)

@pytest.mark.parametrize(
    "idx,val",
    [
        (0, 27.3912),
        (11, 1.8141),
        (16, 1.6535),
    ],
)
def test_parse_dftd4_output(idx, val):
    full_path = full_path = os.path.join(Path(__file__).parent, "reader_data/dftd4.out")
    with open(full_path, "r") as f:
        dftd4_stdout = f.read()
        f.close()

    c6_params = parser.parse_dftd4_output(dftd4_stdout)
    assert c6_params[idx] == val
