from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore
from sklearn.metrics import r2_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset

import mlhess.management.base_settings as globals
from mlhess.machinelearning.architecture.neural_nets import MLH_s


class DummyScheduler:
    def __init__(self) -> None:
        pass

    def step(self):
        pass


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model: nn.Module):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model: nn.Module):
        model.load_state_dict(self.best_model_state)


class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        lr=0.0002,
        gamma=0.99,
        epochs=100,
        model: nn.Module = MLH_s(), #noQA: B008
        criterion: nn.Module = nn.HuberLoss(delta=5e-2),  #noQA: B008
        batch_size: int = 1024,
    ):
        torch.set_num_threads(globals.NUM_THREADS)

        self.device = torch.device(globals.DEVICE)

        if torch.cuda.is_available() and globals.DEVICE == "cuda:0":
            torch.cuda.set_device(self.device)

        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size

        self.epochs = int(epochs)
        self.model: MLH_s = model()
        self.criterion = criterion
        self.is_fitted_ = False

    def fit(self, X, y):
        print(f"Fit procedure is performed on {globals.DEVICE}.")
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        features_train, features_test, targets_train, targets_test = train_test_split(
            X_tensor, y_tensor, train_size=0.99, random_state=24
        )
        # features_train,features_test,targets_train,targets_test = X_tensor,X_tensor,y_tensor,y_tensor
        trainset = TensorDataset(features_train, targets_train)

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of model parameter: {num_params}.\n")

        self.model.update_init_layer(X_tensor.shape[1])
        self.model.update_final_layer(y_tensor.shape[1])

        print(f"The initial layer size is set to: {X_tensor.shape[1]}.")
        print(f"The final layer size is set to: {y_tensor.shape[1]}.")

        self.model = self.model.to(self.device)
        self.model.train()

        criterion = self.criterion
        features_test = features_test.to(self.device)
        n_epochs = self.epochs
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimizer, gamma=self.gamma)
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        r2scores = self.fit_model(
            trainloader,
            features_test,
            targets_test,
            scheduler=scheduler,
            criterion=criterion,
            optimizer=optimizer,
            n_epochs=n_epochs,
        )

        tot_r2scores = r2scores

        np.savetxt("statistics", tot_r2scores)

        self.model = self.model.to(self.device)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray):
        if hasattr(self, "is_fitted_"):
            if not self.is_fitted_:
                raise ValueError("The model instance is not fitted yet.")
        else:
            self.is_fitted_ = True

        self.model.eval()
        self.model.to("cpu")

        with torch.no_grad():
            X_tensor = torch.from_numpy(X).to("cpu").float()

            predictions: torch.Tensor = self.model(X_tensor)

        return predictions.cpu().numpy()

    def fit_model(
        self,
        trainloader: DataLoader,
        features_test: torch.Tensor,
        targets_test: torch.Tensor,
        scheduler=None,
        criterion=None,
        optimizer=None,
        n_epochs=30,
    ) -> np.ndarray:
        statistics = np.zeros([n_epochs, 5])

        t_init = time.time()

        early_stopping = EarlyStopping(patience=25, delta=0.0)

        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        if criterion is None:
            criterion = nn.SmoothL1Loss()
        if scheduler is None:
            scheduler = DummyScheduler()

        for epoch in range(n_epochs):
            self.model.train()

            running_loss = 0.0
            for feature, target in trainloader:
                target = target.to(self.device)
                feature = feature.to(self.device)
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(feature)
                loss = criterion(outputs, target)

                # Backward pass and optimize
                loss.backward()

                optimizer.step()
                running_loss += loss.item()

            with torch.no_grad():
                pred_target = self.model(features_test)
                pred_target = torch.Tensor.cpu(pred_target)

                r2 = r2_score(targets_test.numpy(), pred_target.numpy())
                rmse = np.sqrt(
                    np.mean((pred_target.numpy() - targets_test.numpy()) ** 2)
                )
                val_loss = criterion(pred_target, targets_test).item()

                mae = np.mean(np.abs(pred_target.numpy() - targets_test.numpy()))
                statistics[epoch, 0] = r2
                statistics[epoch, 1] = rmse
                statistics[epoch, 2] = mae
                statistics[epoch, 3] = val_loss
                statistics[epoch, 4] = running_loss / len(trainloader)

            scheduler.step()

            print(
                f"Epoch [{epoch + 1}/{n_epochs}], Loss: {running_loss / len(trainloader):2.4E},  Val: {val_loss:2.4E}"
            )

            early_stopping(val_loss, self.model)

            if early_stopping.early_stop:
                print("Early stopping.")
                break

        early_stopping.load_best_model(self.model)

        t_final = time.time()
        print(f"Fit took {t_final - t_init:3.3f} s")
        return statistics
