from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from mlhess.machinelearning.constructor import Constructor
from mlhess.utils.decorator import initProcess
from mlhess.utils.tools import test_array

if TYPE_CHECKING:
    from mlhess.management.config import Configurator


class TrainingHandler:
    def __init__(self, features, targets, config: Configurator):
        self.config = config
        self.Features = features
        self.Targets = targets

    @initProcess
    def run_protocol(self):
        self.config.internal.train = True

        test_size_threshold = 1e-8

        train_size = self.config.train.train_size

        self.rnd_states = self.config.general.random_state

        self.model_name = self.config.train.model_name

        self.runtype = self.config.general.runtype

        i = None

        if isinstance(train_size, list):
            train_size = sorted(train_size)[::]

            self.shuffle_idx = np.arange(len(self.Features))

            self.shuffle_idx, validation_idx = train_test_split(
                self.shuffle_idx,
                train_size=train_size[-1],
                test_size=self.config.train.validation_size,
                random_state=self.config.general.random_state,
            )

            for i in range(len(train_size)):
                rnd_seed = self.rnd_states

                print(
                    f"Percentage of data set used for training: {train_size[i] * 100} %",
                )

                temp_time_old = time.time()

                train_size_temp = train_size[i] / train_size[-1]
                print(train_size_temp)
                if train_size_temp < 1.0 - 1e-8:
                    shuffle_idx_temp, _ = train_test_split(
                        self.shuffle_idx,
                        train_size=train_size_temp,
                        test_size=self.config.train.validation_size,
                        random_state=self.config.general.random_state,
                    )

                else:
                    shuffle_idx_temp = self.shuffle_idx

                print(f"Total training points:{len(shuffle_idx_temp)}")

                if self.config.train.active:
                    model_trainer = Constructor(self.config.train)
                else:
                    model_trainer = Constructor(self.config.train)

                model_trainer.set_rnd_seed(rnd_seed)
                model_trainer.set_split(i)
                model_trainer.build_model()
                model_trainer.training(
                    features=self.Features,
                    targets=self.Targets,
                    shuffle_idx=shuffle_idx_temp,
                )

                pred_vals = model_trainer.complete_model.predict(
                    self.Features[validation_idx]
                )

                rmsd = np.sqrt(np.mean(self.Targets[validation_idx] - pred_vals) ** 2)

                print("Validation statistics")
                print(
                    f"RndState: {self.config.general.random_state} Size: {train_size[i]} RMSD: {rmsd}"
                )

                if self.config.train.test_size >= test_size_threshold:
                    self.train_size = train_size[i]
                    self.rnd_seed = self.config.general.random_state
                    rmse, mae = test_array(
                        model_trainer.complete_model, self.test_geo, self.config
                    )
                    print("Test statistics")
                    print(
                        f"RndState: {self.config.general.random_state} Size: {train_size[i]} RMSD: {rmse} MAE: {mae}"
                    )

                del model_trainer

        else:
            print(
                f"Percentage of data set used for training: {train_size * 100} %",
            )

            self.shuffle_idx = np.arange(len(self.Features))

            print(len(self.shuffle_idx))

            temp_time_old = time.time()

            if self.config.train.active:
                model_trainer = Constructor(self.config.train)
            else:
                model_trainer = Constructor(self.config.train)

            model_trainer.set_rnd_seed(self.rnd_states)
            model_trainer.set_split(i)
            model_trainer.build_model()
            model_trainer.training(
                features=self.Features,
                targets=self.Targets,
                shuffle_idx=self.shuffle_idx,
            )

            self.score = model_trainer.score

            y_pred = model_trainer.complete_model.predict(
                self.Features[self.shuffle_idx]
            )
            print(r2_score(y_true=self.Targets[self.shuffle_idx], y_pred=y_pred))

            temp_time_new = time.time()

            print(f"Training was done in {round(temp_time_new - temp_time_old)} s")

            if (
                self.config.train.test_size >= test_size_threshold
                and len(self.test_geo) > 0
            ):
                temp_time_old = time.time()

                self.train_size = self.config.train.test_size
                self.rnd_seed = self.config.general.random_state

                rmse, mae = test_array(
                    model_trainer.complete_model, self.test_geo, self.config
                )
                print("Test statistics")
                print(
                    f"RndState: {self.config.general.random_state} Size: {train_size[i]} RMSD: {rmse} MAE: {mae}"
                )

                temp_time_new = time.time()

                print(
                    f"Testing was done in {temp_time_new - temp_time_old: 0.2f} s",
                )
