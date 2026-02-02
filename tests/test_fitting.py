import os
import torch
import numpy as np

from pathlib import Path

from mlhess.scripts.data_handling import FittingDataHandler
from mlhess.scripts.training_handler import TrainingHandler
from mlhess.management.config import Configurator
import mlhess.management.base_settings as base_settings
from mlhess.machinelearning.architecture.base_model import MachineLearningModel
from mlhess.machinelearning.architecture.neural_nets import MLH_s
from sklearn.preprocessing import StandardScaler


def test_xy():
    config_path = os.path.join(Path(__file__).parent, "fitting_data/input.toml")
    config = Configurator(config_path)
    base_settings.PROCESSED_DATA_FOLDER = os.path.join(
        Path(__file__).parent, "fitting_data/processed_data"
    )

    random_state = config.general.random_state
    torch.random.manual_seed(random_state)
    np.random.seed(random_state)

    fdh = FittingDataHandler(config)
    fdh.run_protocol()

    trainer = TrainingHandler(fdh.Features, fdh.Targets, config)
    trainer.run_protocol()

    model = MachineLearningModel(config)
    model.scaler = StandardScaler()
    model.model = MLH_s
    model.fit(fdh.Features[trainer.shuffle_idx], fdh.Targets[trainer.shuffle_idx])

    assert np.round(trainer.score, 3) == np.round(0.1388, 3)
    assert np.round(
        model.score(
            fdh.Features[trainer.shuffle_idx], fdh.Targets[trainer.shuffle_idx]
        ),
        3,
    ) > np.round(0.1388, 3)
