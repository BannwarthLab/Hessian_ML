#!/usr/bin/env python3
from __future__ import annotations

import torch

import time
import numpy as np

# from tblite.interface import Calculator
from mlhess.management.config import Configurator
from mlhess.scripts.data_handling import FittingDataHandler
from mlhess.scripts.training_handler import TrainingHandler
from mlhess.scripts.predicting_handler import PredictionHandling
import mlhess.utils.io.parser as parser

# has to be beneath tblite to circumvent conflicts


def main() -> None:
    init_time = time.time()
    arg_parser = parser.parse_cmd_line_input()
    arguments = arg_parser.parse_args()

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
