from __future__ import annotations

import os
import sys
import tomli
from argparse import ArgumentParser
import numpy as np


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlhess.management.config import Configurator


def parse_cmd_line_input() -> ArgumentParser:
    """Parses through the command line input.

    :return: ArugmentParser class
    :rtype: ArgumentParser
    """
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="default",
        help="INPUT must be a .toml file",
    )

    arg_parser.add_argument(
        "-to",
        "--toml",
        action="store_true",
        help="generates a basic example .toml file",
    )
    return arg_parser


def parse_input_toml_file(input_file_name: str) -> dict:
    """Parser for the .toml to generate a dictionoary for the input information.

    :param input_file_name: name of the input file
    :type input_file_name: str
    :return: All information from the toml-file.
    :rtype: dict
    """

    inp_file = input_file_name

    if os.path.isfile(inp_file):
        with open(inp_file, mode="rb") as fp:
            parsed_config = tomli.load(fp)
            print(f"Input file: {inp_file} is used.")

    elif inp_file == "default":
        if os.path.isfile("input.toml"):
            with open("input.toml", mode="rb") as fp:
                parsed_config = tomli.load(fp)

                print("Input file: input.toml is used.")
        else:
            print("Default setup is used.")

            parsed_config = {}  # Path(__file__).parent / "default_input/input.toml"

            # with open(abs_path, mode="rb") as fp:
            #     parsed_config = tomli.load(fp)

    else:
        print(f"Input file: {inp_file} not found.")
        print("")

        sys.exit(1)

    return parsed_config


def parse_data_set(config: Configurator) -> list:
    """Search for all folders which include a xyzfile and optionally a target_file.

    :param config: overall configuration
    :type config: Configurator
    :return: found folders
    :rtype: list
    """
    main_folder = config.collector.folder
    if os.path.isdir(main_folder):
        msg = f"Parsing data set in {main_folder}..."
        print(msg, end="")

        file_subset = set([config.molecule.xyz_file])
        if config.general.collector:
            file_subset.add(config.molecule.target_file)

        walker = os.walk(main_folder)

        folders = []
        for folder, _, files in walker:
            if file_subset.issubset(files):
                folders.append(folder)

        folders = sorted(folders)

        print("done")

        msg = f"Total of {len(folders)} folders found."
        print("")
        print(msg)
        print("")

        return folders
    return []


def parse_dftd4_output(output: str) -> np.ndarray:
    """Get the C6 parameter from the DFT-D4 output.

    :param output: output of the dftd4 program
    :type output: str
    :return: C6 parameter
    :rtype: np.ndarray
    """
    lines = output.split("\n")

    for idx, line in enumerate(lines):
        if "Atomic properties (in atomic units):" in line:
            break

    init_idx = idx + 4

    for idx, line in enumerate(lines[init_idx:]):
        if "---" in line:
            break

    final_idx = idx + init_idx

    c6_params = []
    for line in lines[init_idx:final_idx]:
        c6_params.append(line.split()[-2])

    return np.array(c6_params, dtype=np.float64)
