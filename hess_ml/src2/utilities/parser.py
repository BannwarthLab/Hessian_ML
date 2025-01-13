from __future__ import annotations

import os
import sys
import tomli 
import numpy as np 
from argparse import ArgumentParser
from pathlib import Path

class Parser:
    def __init__(self) -> None:
        self.cwd = os.getcwd()
        # self.LoadDefaultConfig()

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

    def parse_toml(self) -> None:
        """

        Parser for the .toml to generate a dictionoary for the input information.

        """

        inp_file = self.arg_parser.parse_args().input

        if os.path.isfile(inp_file):
            with open(self.arg_parser.parse_args().input, mode="rb") as fp:
                self.parsed_config = tomli.load(fp)
                print(f"Input file: {inp_file} is used.")

        elif inp_file == "default":
            if os.path.isfile("input.toml"):
                with open("input.toml", mode="rb") as fp:
                    self.parsed_config = tomli.load(fp)

                    print("Input file: input.toml is used.")
            else:
                print("Default setup is used.")

                abs_path = Path(__file__).parent / "default_input/input.toml"

                with open(abs_path, mode="rb") as fp:
                    self.parsed_config = tomli.load(fp)

        else:
            print(f"Input file: {inp_file} not found.")
            print("")

            sys.exit()

    def get_config(self) -> dict:
        return self.parsed_config

    def parse_data_set(self, main_folder) -> None:
        print(f"Parsing data set in {main_folder}...", end="")

        xyz_file = self.parsed_config.get("molecule", {"xyz_file": "xtbopt.xyz"}).get(
            "xyz_file",
        )
        target_file = self.parsed_config.get("molecule", {"target_file": "hessian"}).get(
            "target_file",
        )

        walker = os.walk(main_folder)

        self.folders = []

        if self.config.general.feature:
            for folder,_,files in walker:
                if xyz_file in files and target_file in files:
                    self.folders.append(folder)
        else:
            for folder,_,files in walker:
                if xyz_file in files:
                    self.folders.append(folder)
        self.folders = sorted(self.folders)
        

        print("done")

        if len(self.folders) == 0:
            print("No folders were found. Stopping the program.")
            sys.exit()
            
        print(f"Total of {len(self.folders)} folders found.")
        print("")



def parse_dftd4_output(output:str)-> np.ndarray:
    """Get the C6 parameter from the DFT-D4 output.

    :param output: output of the dftd4 program 
    :type output: str
    :return: C6 parameter
    :rtype: np.ndarray
    """
    lines = output.split("\n")

    for idx, line in enumerate(lines):
        if  "Atomic properties (in atomic units):" in line:
            break

    init_idx = idx+4 

    for idx,line in enumerate(lines[init_idx:]):
        if "---" in line:
            break

    final_idx = idx+init_idx

    c6_params = []
    for line in lines[init_idx:final_idx]:
        c6_params.append(line.split()[-2])
        
    return np.array(c6_params,dtype=np.float64)