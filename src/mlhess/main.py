"""Programme entry point for the Machine Learning Hessian package.

This module can be used to collect data, train models, and predict new targets.
"""
from __future__ import annotations

import os
import time

import numpy as np
import sklearn  #noQA: F401
import torch

# tblite, torch, and sklearn each bundle their own libomp.dylib on macOS;
# this must be set before any of them are imported below, or the process
# aborts with "OMP: Error #15".
#This import has to be here otherwise errors may occur.
from tblite.interface import Calculator  # noqa: F401

from mlhess.management.config import Configurator
from mlhess.scripts.compute_xyz import comp_hessian_from_xyz_cli
from mlhess.scripts.data_handling import FittingDataHandler
from mlhess.scripts.predicting_handler import PredictionHandling
from mlhess.scripts.training_handler import TrainingHandler
from mlhess.utils.io import parser

# has to be beneath tblite to circumvent conflicts
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

def main() -> None:
    """
    Entry point for programme.
    """
    init_time = time.time()
    arg_parser = parser.parse_cmd_line_input()
    arguments = arg_parser.parse_args()


    if arguments.xyz is not None:
        comp_hessian_from_xyz_cli(arguments.xyz)

    else:

        config = Configurator(arguments.input)

        random_state = config.general.random_state
        torch.random.manual_seed(random_state)
        np.random.seed(random_state)

        if config.general.collector:
            fit_data_handler = FittingDataHandler(config)
            config = fit_data_handler.run_protocol()

        if config.general.trainer:
            training_handler = TrainingHandler(
                fit_data_handler.Features, fit_data_handler.Targets, config
            )
            training_handler.run_protocol()

        if config.general.predictor:
            prediction_handler = PredictionHandling(config)
            prediction_handler.run_protocol()

    msg = f"Total wall time: {time.time() - init_time:4.2F} s"
    print(msg)


if __name__ == "main":
    main()
