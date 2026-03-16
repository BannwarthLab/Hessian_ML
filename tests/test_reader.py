import os
import pytest
import numpy as np

from mlhess.utils.io.reader import read_txt_file, read_dftd4
from pathlib import Path


@pytest.mark.parametrize(
    "test_data_path, expected",
    [
        ("sample_files.txt", 6),
        ("does_not_exist.txt", 0),
    ],
)
def test_read_txt_file(test_data_path, expected):
    full_path = os.path.join(Path(__file__).parent, "reader_data", test_data_path)
    lines = read_txt_file(full_path)
    assert type(lines) is list
    assert len(lines) == expected


def test_read_dftd4():
    test_data_path = "dftd4.out"
    full_path = os.path.join(Path(__file__).parent, "reader_data", test_data_path)
    c6_params = read_dftd4(full_path)
    c6_params_true = np.array(
        [
            27.3912,
            27.0125,
            21.7983,
            25.7308,
            25.5191,
            33.4696,
            35.4200,
            1.8197,
            1.6095,
            1.8308,
            1.7781,
            1.8141,
            1.5824,
            1.5691,
            1.6457,
            1.6258,
            1.6535,
        ]
    )
    np.testing.assert_array_equal(c6_params, c6_params_true)
