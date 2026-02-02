import os
import pytest

from mlhess.utils.io.reader import read_txt_file
from mlhess.utils.io.writer import list_to_txt, truncate_file
from pathlib import Path


@pytest.mark.parametrize(
    "test_data_path, expected",
    [
        ("write_file.txt", ["a", "b", "c"]),
    ],
)
def test_list_txt_file(test_data_path, expected):
    full_path = os.path.join(Path(__file__).parent, "writer_data", test_data_path)

    list_to_txt(expected, full_path)

    lines = read_txt_file(full_path)

    for line, expected_line in zip(lines, expected):
        assert line == expected_line


@pytest.mark.parametrize(
    "test_data_path",
    [
        ("write_file.txt"),
    ],
)
def test_truncate_file(test_data_path):
    full_path = os.path.join(Path(__file__).parent, "writer_data", test_data_path)

    list_to_txt(["a", "b"], full_path)

    truncate_file(full_path)

    with open(full_path, "r") as f:
        lines = f.readlines()

    assert lines == []
