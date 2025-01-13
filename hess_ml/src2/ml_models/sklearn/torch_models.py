import torch
import time as time 
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import DataLoader,TensorDataset
import hess_ml.src2.governance.globals as globals
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
from torch.optim.lr_scheduler import ExponentialLR


class DummyScheduler():
    
    def __init__(self) -> None:
        pass

    def step(self):
        pass 


class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        layer_size = 350
        self.init_layer_size = 1

        self.fc1 = nn.Linear(self.init_layer_size, layer_size)  
        self.norm1 = nn.LayerNorm(layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size) 
        self.fc3 = nn.Linear(layer_size, layer_size)
        #self.drop2 = nn.Dropout(p=0.5)
        self.fc31 = nn.Linear(layer_size, layer_size)  
        #self.norm2 = nn.LayerNorm(layer_size)
        self.fc32 = nn.Linear(layer_size, layer_size)  
        self.fc33 = nn.Linear(layer_size, layer_size,bias=False)  

        self.norm3 = nn.LayerNorm(layer_size)
        self.fc42 = nn.Linear(layer_size, layer_size,bias=False)  
        self.fc43 = nn.Linear(layer_size, layer_size,bias=False)  
        
        self.fc5 = nn.Linear(layer_size, 64,bias=False) 
        self.fc_last = nn.Linear(64, 9,bias=False) 

    def update_init_layer(self,size:int):
        self.fc1 = nn.Linear(size, 350)
        self.init_layer_size = size

    def forward(self, x):
        x = x.view(-1, self.init_layer_size)  # Flatten the input
        x = torch.nn.functional.tanh(self.fc1(x))
        x = torch.nn.functional.tanh(self.fc2(x))
        x = torch.nn.functional.hardtanh(self.fc3(x))
        x = self.norm1(x)
        #x = self.drop2(x)
        x = torch.nn.functional.tanh(self.fc31(x))
        
        #x = self.norm2(x)
        #x = torch.nn.functional.tanh(self.fc32(x))
        x = torch.nn.functional.tanh(self.fc32(x))
        x = torch.nn.functional.hardtanh(self.fc33(x))
        
        x = self.norm3(x)
        x = torch.nn.functional.tanh(self.fc42(x))
        x = torch.nn.functional.hardtanh(self.fc43(x))

        x = torch.nn.functional.tanh(self.fc5(x))

        x = self.fc_last(x)

        return x
    
# Define a simple PyTorch model
class HessiaNN(nn.Module):
    def __init__(self):
        super(HessiaNN, self).__init__()
        self.init_layer_size = 1
        layer_size = 700
        self.fc1 = nn.Linear(self.init_layer_size, layer_size) 
        self.norm = nn.LayerNorm(layer_size,eps=5e-7)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, layer_size)
        self.fc_last = nn.Linear(layer_size, 9)

    def update_init_layer(self,size:int):
        self.fc1 = nn.Linear(size, 700)
        self.init_layer_size = size

    def forward(self, x):
        x = x.view(-1, self.init_layer_size)  # Flatten the input
        x = nn.functional.tanhshrink(self.fc1(x))
        x = self.norm(x)
        x = nn.functional.tanhshrink(self.fc2(x))
        x = nn.functional.tanhshrink(self.fc3(x))
        x = nn.functional.softshrink(self.fc_last(x),lambd=1e-6)
        return x

class CustomNN(nn.Module):
    def __init__(self):
        super(CustomNN, self).__init__()
        self.init_layer_size = 1
        layer_size = 700
        self.fc1 = nn.Linear(self.init_layer_size,layer_size) 
        self.norm = nn.LayerNorm(layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, layer_size)
        self.fc4 = nn.Linear(layer_size, layer_size,bias=False)
        self.fc_last = nn.Linear(layer_size, 9, bias=False)

    def update_init_layer(self,size:int):
        self.fc1 = nn.Linear(size, 700)
        self.init_layer_size = size

    def forward(self, x):
        x = x.view(-1, self.init_layer_size)  # Flatten the input
        x = nn.functional.softshrink(self.fc1(x),lambd=1e-7)
        x = self.norm(x)
        x = nn.functional.tanhshrink(self.fc2(x))
        x = nn.functional.tanh(self.fc3(x))
        x = nn.functional.tanhshrink(self.fc4(x))
        x = nn.functional.hardshrink(self.fc_last(x),lambd=1e-5)

        return x

class EnlargedNN(nn.Module):
    def __init__(self):
        super(EnlargedNN, self).__init__()
        self.init_layer_size = 1
        layer_size = 900
        self.fc1 = nn.Linear(self.init_layer_size,layer_size) 
        self.norm = nn.LayerNorm(layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, layer_size)
        self.fc4 = nn.Linear(layer_size, layer_size,bias=False)
        self.fc_last = nn.Linear(layer_size, 9, bias=False)

    def update_init_layer(self,size:int):
        self.fc1 = nn.Linear(size, 900)
        self.init_layer_size = size

    def forward(self, x):
        x = x.view(-1, self.init_layer_size)  # Flatten the input
        x = nn.functional.softshrink(self.fc1(x),lambd=1e-7)
        x = self.norm(x)
        x = nn.functional.tanhshrink(self.fc2(x))
        x = nn.functional.tanh(self.fc3(x))
        x = nn.functional.tanhshrink(self.fc4(x))
        x = nn.functional.hardshrink(self.fc_last(x),lambd=1e-5)

        return x

class HessiaNN_custom(nn.Module):
    def __init__(self):
        super(HessiaNN_custom, self).__init__()
        self.init_layer_size = 1
        layer_size = 700
        self.fc1 = nn.Linear(self.init_layer_size,layer_size) 
        self.norm = nn.LayerNorm(layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, layer_size)
        self.fc4 = nn.Linear(layer_size, layer_size,bias=False)
        self.fc_last = nn.Linear(layer_size, 9, bias=False)

    def update_init_layer(self,size:int):
        self.fc1 = nn.Linear(size, 700)
        self.init_layer_size = size

    def forward(self, x):
        x = x.view(-1, self.init_layer_size)  # Flatten the input
        x = nn.functional.softshrink(self.fc1(x),lambd=1e-7)
        x = self.norm(x)
        x = nn.functional.tanhshrink(self.fc2(x))
        x = nn.functional.tanh(self.fc3(x))
        x = nn.functional.tanhshrink(self.fc4(x))
        x = nn.functional.hardshrink(self.fc_last(x),lambd=1e-5)

        return x

class ReducedNN(nn.Module):
    def __init__(self):
        super(ReducedNN, self).__init__()
        self.init_layer_size = 1
        layer_size = 400
        self.fc1 = nn.Linear(self.init_layer_size, layer_size)  
        self.norm1 = nn.LayerNorm(layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size) 
        self.fc3 = nn.Linear(layer_size, layer_size)  
        self.fc31 = nn.Linear(layer_size, layer_size)  
        self.norm2 = nn.LayerNorm(layer_size)
        self.fc32 = nn.Linear(layer_size, layer_size)  
        self.fc33 = nn.Linear(layer_size, layer_size)  

        self.norm3 = nn.LayerNorm(layer_size)
        self.fc42 = nn.Linear(layer_size, layer_size,bias=False)  
        self.fc43 = nn.Linear(layer_size, layer_size,bias=False)  
        
        self.fc5 = nn.Linear(layer_size, 64,bias=False) 
        self.fc_last = nn.Linear(64, 9) 

    def update_init_layer(self,size:int):
        self.fc1 = nn.Linear(size, 400)
        self.init_layer_size = size

    def forward(self, x):
        x = x.view(-1, self.init_layer_size)  # Flatten the input
        x = torch.nn.functional.tanh(self.fc1(x))
        x = self.norm1(x)
        x = torch.nn.functional.tanh(self.fc2(x))
        x = torch.nn.functional.hardtanh(self.fc3(x))
        x = torch.nn.functional.sigmoid(self.fc31(x))

        x = self.norm2(x)
        x = torch.nn.functional.tanhshrink(self.fc32(x))
        x = torch.nn.functional.hardtanh(self.fc33(x))

        x = self.norm3(x)
        x = torch.nn.functional.sigmoid(self.fc42(x))
        x = torch.nn.functional.hardtanh(self.fc43(x))

        x = torch.nn.functional.tanh(self.fc5(x))

        x = self.fc_last(x)
        return x

class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, lr=0.0002, gamma=0.99, epochs=100,model:nn.Module=HessiaNN(),criterion:nn.Module=nn.HuberLoss(delta=5e-2),
                 batch_size:int=1024):
    
        torch.set_num_threads(globals.NUM_THREADS)

        self.device  = torch.device(globals.DEVICE)

        if torch.cuda.is_available() and globals.DEVICE=="cuda:0":
            torch.cuda.set_device(self.device)
        
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size

        self.epochs = int(epochs)
        self.model:HessiaNN = model
        self.criterion = criterion
        self.is_fitted_ = False

    def fit(self, X, y):
        print(f"Fit procedure is performed on {globals.DEVICE}.")
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        features_train,features_test,targets_train,targets_test = train_test_split(X_tensor,y_tensor,train_size=0.99,random_state=24)
        trainset  = TensorDataset(features_train,targets_train)

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of model parameter: {num_params}.\n")

        self.model.update_init_layer(X_tensor.shape[1])
        print(f"The initial layer size is set to: {X_tensor.shape[1]}.")

        self.model = self.model.to(self.device)
        self.model.train()

        criterion = nn.HuberLoss(delta=5e-2)
        features_test = features_test.to(self.device)
        n_epochs = 20
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = ExponentialLR(optimizer,gamma=0.95)
        trainloader = DataLoader(trainset , batch_size=self.batch_size, shuffle=True)
        r2scores = self.fit_model(trainloader,features_test,targets_test,scheduler=scheduler,
                                  criterion=criterion,
                                  optimizer=optimizer,
                                  n_epochs=n_epochs)
        
        tot_r2scores = r2scores

        self.model = self.model.to(self.device)
        self.model.train()
        
        criterion = self.criterion
        features_test = features_test.to(self.device)
        n_epochs = self.epochs
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimizer,gamma=self.gamma)
        trainloader = DataLoader(trainset , batch_size=self.batch_size, shuffle=True)
        r2scores = self.fit_model(trainloader,features_test,targets_test,scheduler=scheduler,
                                  criterion=criterion,
                                  optimizer=optimizer,
                                  n_epochs=n_epochs)

        tot_r2scores = np.append(tot_r2scores,r2scores,axis=0)

        np.savetxt('statistics',tot_r2scores)

        self.model = self.model.to(self.device)
        self.is_fitted_ = True 

        return self

    def predict(self, X:np.ndarray):

        if hasattr(self, "is_fitted_"):
            if not self.is_fitted_:
                raise ValueError("The model instance is not fitted yet.")
        else:
            self.is_fitted_ = True

        self.model.eval()
        self.model.to("cpu")

        with torch.no_grad():
            X_tensor = torch.from_numpy(X).to("cpu")
            predictions:torch.Tensor = self.model(X_tensor)

        return predictions.cpu().numpy()

    def fit_model(self,trainloader:TensorDataset,
                  features_test:torch.Tensor,
                  targets_test:torch.Tensor,
                  scheduler=None,
                  criterion=None,
                  optimizer=None,
                  n_epochs=30) -> np.ndarray:

        statistics = np.zeros([n_epochs,4])

        t_init = time.time()

        if optimizer is None:
            optim.Adam(self.model.parameters(), lr=0.001)
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

                r2 = r2_score(pred_target.numpy(),targets_test.numpy())
                rmse = np.sqrt(np.mean((pred_target.numpy()-targets_test.numpy())**2))
                loss = criterion(pred_target,targets_test)
                mae = np.mean(np.abs(pred_target.numpy()-targets_test.numpy()))

                statistics[epoch,0] = r2
                statistics[epoch,1] = rmse 
                statistics[epoch,2] = mae
                statistics[epoch,3] = loss.item()

            scheduler.step()

            print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {running_loss / len(trainloader):2.4E}')

        t_final = time.time()
        print(f'Fit took {t_final-t_init:3.3f} s')
        return statistics
